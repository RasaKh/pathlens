# PathLENS (Shortcut Pathology)

Training, evaluation, attribution and analysis code for Shortcut Pathology: a study of how pathology classifiers score well while relying on non-biological cues, and how performance changes when held-out splits move from patches to slides, patients and hospitals.

The live walkthrough and interactive demo run at [pathlens.khosrofar.me](https://pathlens.khosrofar.me). This repository holds the modeling and analysis code; the Flask serving app is deployed separately.

## Study phases

- Phase 1 (PathMNIST, 9-class colorectal tissue classification at 64 px): stamps a deterministic 9x9 ring glyph onto training images of colorectal adenocarcinoma epithelium (class 8). Two `PatchNet` CNNs with the same architecture and seed are trained on CPU (one on clean images, one on stamped images) to measure how the ring shifts predictions, Grad-CAM maps and Integrated Gradients maps.
- Phase 2 (Camelyon17-WILDS, binary lymph-node tumor detection across five hospitals): builds deterministic five-fold splits at four levels (`patch`, `slide`, `patient`, `hospital`) and trains plain and stain-augmented `PatchNet` models across seeds (`configs/phase2_gpu.yaml`). A second pass computes Ruifrok-Johnston HED stain statistics, fits hospital-prediction probes on learned features, and builds a UMAP embedding map and neighbour index for a seeded ~126,000-patch analysis subset out of 455,954 patches.
- Phase 3 (frozen `owkin/phikon` foundation model): extracts 768-dim CLS features from `owkin/phikon` on the same Phase 2 subset patches and fits linear heads with slide-grouped inner cross-validation across all four held-out levels. It also tests whether projecting out the hospital-predictive direction changes accuracy, checks how a decision threshold tuned on four hospitals transfers to the fifth, and computes 95% slide-cluster bootstrap intervals (`B = 1000`).

## Environment setup

The project targets Python 3.14 and defines four `uv` extras in `pyproject.toml`:

- `dev`: CPU PyTorch, torchvision, medmnist, wilds, scikit-image, umap-learn, `pytest`, `pytest-timeout` and `ruff` (use this for Phase 1, Phase 2/3 analysis and tests).
- `analysis`: same as `dev` without test and lint tools.
- `train`: CUDA 12.8+ PyTorch (`pytorch-cu128` index) plus pandas, pyarrow and scikit-learn for GPU training runs. In `pyproject.toml`, `train` conflicts with the CPU extras (`serve`, `analysis`, `dev`), so install it in its own environment on the GPU machine.
- `serve`: CPU PyTorch and Flask/Gunicorn dependencies shared with the web deployment.

```bash
uv sync --extra dev          # CPU analysis, Phase 1 and tests
# or on a CUDA 12.8+ machine:
uv sync --extra train        # Phase 2 GPU training
```

Exported requirement snapshots for each extra are also checked in (`requirements-dev.txt`, `requirements-analysis.txt`, `requirements-train.txt`, `requirements-serve.txt`).

Phase 3 feature extraction (`scripts/extract_p3.py`) does not import `pathlens` and runs standalone on a GPU machine with `torch`, `transformers` and `numpy` installed. It downloads `owkin/phikon` from Hugging Face.

All datasets, run directories and output artifacts resolve through `pathlens/paths.py` under `$PATHLENS_HOME`. The default root is `<repo>/workdir` (git-ignored). Each run writes its config, training log, weights, predictions, metrics and a completion marker (`DONE`) into a folder named by its deterministic run id (for example `p2-hospital-f2-aug-s0`).

## Commands

Phase 1 (PathMNIST, CPU; uses the manually downloaded PathMNIST `.npz` archive under `$PATHLENS_HOME`):

```bash
uv run python -m scripts.prepare_p1 --sample-limit 2000   # smoke set under $PATHLENS_HOME/data/p1-smoke/
uv run python -m scripts.prepare_p1                       # full 64 px set under $PATHLENS_HOME/data/p1/
uv run python -m scripts.train_p1 --variant clean --seed 0
uv run python -m scripts.train_p1 --variant poisoned --seed 0
uv run python -m scripts.eval_p1 --seed 0
```

Phase 2 (Camelyon17-WILDS; `--src` points to a local Camelyon17-WILDS download directory or archive, or use `scripts.fetch_subset_p2` to stream only the ~126,000 analysis subset patches):

```bash
uv run python -m scripts.prepare_p2 --src /path/to/camelyon17   # unpack Camelyon17 patches into memmap arrays
uv run python -m scripts.inspect_metadata                       # summarise slide, patient and hospital hierarchy
uv run python -m scripts.make_splits --levels 4 --seed 0        # write deterministic 5-fold splits
uv run python -m pathlens.run_all --config configs/phase2_gpu.yaml --probe-only  # benchmark GPU speed vs time budget
uv run python -m pathlens.run_all --config configs/phase2_gpu.yaml               # run GPU training queue and pack archive
uv run python -m scripts.verify_artifacts                       # verify packed Phase 2 archive checksums
uv run python -m scripts.aggregate_p2                           # aggregate metrics and bootstrap intervals
uv run python -m scripts.probes_and_map                         # hospital probes, UMAP map and neighbour index
uv run python -m scripts.stain_stats                            # per-hospital HED stain statistics
uv run python -m scripts.curate                                 # select seeded example patches for the walkthrough
```

Phase 3 (frozen Phikon comparison; `analyse_p3` defaults to `$PATHLENS_HOME/p3/p3_phikon.npy` if `--features` is omitted):

```bash
python scripts/extract_p3.py --data /path/to/subset --out /path/to/p3   # standalone GPU extractor (needs transformers)
uv run python -m scripts.analyse_p3 fit --jobs 2                        # fit linear heads across all four levels
uv run python -m scripts.analyse_p3 stats --jobs 2                      # bootstrap intervals, probes and threshold transfer
```

Unit and integration tests (offline, CPU only, no external datasets needed):

```bash
uv run pytest
```

## Repository layout

```text
configs/
  phase2_gpu.yaml       training queue, PatchNet widths [32, 64, 128, 256] and GPU time budget
pathlens/
  model.py              hand-written ResNet-style CNN (PatchNet)
  attribution.py        Grad-CAM and chunked Integrated Gradients
  stain.py              Ruifrok-Johnston HED colour deconvolution and stain transforms
  stamp.py              deterministic 9x9 ring shortcut stamp
  splits.py             patch, slide, patient and hospital splits
  train.py, run_all.py  training loop and GPU queue runner with zstd archive packing
  evaluate.py           metrics and bootstrap intervals
  probes.py, embed.py   hospital probes, UMAP map and neighbour index
  fm.py                 Phase 3 linear heads, hospital-direction projection and threshold transfer
  result_schemas.py     schema and finiteness checks for result JSON files
  paths.py, config.py   PATHLENS_HOME layout and frozen run configs
scripts/                CLI entry points for preparation, training, analysis and verification
tests/                  unit and integration test suite
```

## Scope

This repository is a research and educational codebase built to inspect shortcut learning and site shift in pathology images. It is not a diagnostic tool.
