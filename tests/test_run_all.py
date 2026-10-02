"""pathlens.run_all and scripts/verify_artifacts.py on a tiny fake Phase 2 dataset, on the CPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pathlens import paths, run_all
from pathlens.config import expand_queue
from pathlens.identity import run_id
from pathlens.splits import make_splits

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("verify_artifacts", ROOT / "scripts" / "verify_artifacts.py")
va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(va)

YAML = """\
budget_hours: {budget}
probe_steps: 3
allow_partial: {partial}
model: {{widths: [4, 8, 8, 8], num_classes: 2}}
optim: {{epochs: {epochs}, batch_size: 8, peak_lr: 3.0e-3}}
compile: false
amp_dtype: fp32
queue:
  - {{levels: [hospital, patch], folds: [2, 0], variant: plain, seeds: [0]}}
  - {{levels: [hospital], folds: [2], variant: aug, seeds: [0]}}
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_home(base: Path, levels: int = 4) -> Path:
    """base/data/p2/{patches_u8.npy, meta.parquet, splits.parquet, data_manifest.json}: 5 hospitals x 10 slides x 3 patches."""
    p2 = base / "data" / "p2"
    p2.mkdir(parents=True)
    rows = []
    slide = 0
    for h in range(5):
        for s in range(10):
            patient = h * 10 + s // 2
            for k in range(3):
                rows.append({"row": len(rows), "wilds_idx": len(rows), "hospital": h, "patient": patient, "node": s % 2,
                             "slide": slide, "x": k, "y": s, "label": (s + k) % 2, "wilds_split": "train",
                             "slide_key": f"h{h}-p{patient:03d}-n{s % 2}-s{s}"})
            slide += 1
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    np.save(p2 / "patches_u8.npy", rng.integers(1, 256, (len(meta), 96, 96, 3), dtype=np.uint8))
    meta.to_parquet(p2 / "meta.parquet")
    make_splits(meta, 5, 0, levels).to_parquet(p2 / "splits.parquet")
    files = {n: {"sha256": sha(p2 / n)} for n in ("patches_u8.npy", "meta.parquet", "splits.parquet")}
    files["splits.parquet"]["levels_present"] = levels
    (p2 / "data_manifest.json").write_text(json.dumps({"files": files}))
    return base


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.home.cache_clear()
    make_home(tmp_path)
    yield tmp_path
    paths.home.cache_clear()


def cfg_file(tmp_path, budget=4, partial="false", epochs=1) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(YAML.format(budget=budget, partial=partial, epochs=epochs))
    return p


# ------------------------------------------------------------------ the queue

def test_the_real_config_expands_to_the_docs_35_runs_in_priority_order(tmp_path):
    runs = expand_queue(ROOT / "configs" / "phase2_gpu.yaml", levels_present=4)
    ids = [run_id(r) for r in runs]
    assert len(runs) == 35 and len(set(ids)) == 35
    assert ids[:5] == ["p2-hospital-f2-plain-s0", "p2-hospital-f0-plain-s0", "p2-hospital-f1-plain-s0",
                       "p2-hospital-f3-plain-s0", "p2-hospital-f4-plain-s0"]
    assert ids[5].startswith("p2-patch-f2-plain") and ids[10].startswith("p2-slide-f2") and ids[15].startswith("p2-patient-f2")
    assert ids[20] == "p2-hospital-f2-aug-s0" and all("-s1" in i for i in ids[25:])
    assert len(expand_queue(ROOT / "configs" / "phase2_gpu.yaml", levels_present=3)) == 30


def test_levels_present_comes_from_the_data_manifest(tmp_path):
    base = tmp_path / "h3"
    make_home(base, levels=3)
    cfg = run_all.load_session(ROOT / "configs" / "phase2_gpu.yaml", base / "data" / "p2")
    assert len(cfg.runs) == 30 and not any(r.level == "patient" for r in cfg.runs) and cfg.min_gpu_gb == 14.0


def test_estimate_formula():
    cfg = expand_queue(ROOT / "configs" / "phase2_gpu.yaml", 4)[0]
    assert run_all.estimate_seconds(cfg, 300_000, 90_000, 1_000.0) == pytest.approx(6 * 300 + 30)


# ------------------------------------------------------------------ the whole flow, dry

def test_dry_run_end_to_end_then_skip_then_verify(env):
    cfg = cfg_file(env)
    assert run_all.run_session(cfg, dry_run=True) == run_all.EXIT_OK
    runs = env / "runs_dry"
    q = json.loads((runs / "queue.json").read_text())
    assert len(q) == 5 and q[0]["fold"] == 2 and q[0]["level"] == "hospital"
    for r in q:
        d = runs / f"p2-{r['level']}-f{r['fold']}-{r['variant']}-s0"
        assert (d / "DONE").exists() and not (d / "ckpt.pt").exists()
    preds = pd.read_parquet(runs / "p2-hospital-f2-plain-s0" / "predictions.parquet")
    assert {"row", "label", "pred", "p_tumour", "hospital", "slide_key", "patient"} <= set(preds.columns)
    assert set(preds["hospital"]) == {2}                                   # the held-out hospital only
    assert preds["slide_key"].str.startswith("h2-").all()

    a = env / "analysis_inputs_dry"
    for name in ("plain", "aug"):
        m, pr = np.load(a / f"map_{name}.npy"), np.load(a / f"probe_{name}.npy")
        assert m.dtype == pr.dtype == np.float16 and m.shape[1] == pr.shape[1] == 8
        assert np.allclose(np.linalg.norm(m.astype(np.float32), axis=1), 1.0, atol=2e-3)
    mr, pr_rows = np.load(a / "map_rows.npy"), np.load(a / "probe_rows.npy")
    assert len(mr) == 40 and not set(mr) & set(pr_rows)                    # 5 hospitals x 2 labels x 4

    man = json.loads((runs / "session_manifest.json").read_text())
    assert man["dry_run"] is True and man["failed_runs"] == [] and man["probe"]["images_per_second"] > 0
    assert "runs_dry/p2-hospital-f2-plain-s0/model.pt" in man["files"]
    assert not any(k.endswith("ckpt.pt") for k in man["files"])

    archive = env / "artifacts_phase2_dry.tar.zst"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert "packed; sha256 " + digest in (runs / "session.log").read_text()
    assert run_all.run_session(cfg, dry_run=True) == run_all.EXIT_OK        # second start: everything is skipped
    assert (runs / "session.log").read_text().count("already done, skipped") == 5


def test_verify_artifacts_accepts_the_pack_and_rejects_damage(env, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "MANIFEST", "runs_dry/session_manifest.json")      # the dry run keeps its files under runs_dry/
    assert run_all.run_session(cfg_file(env), dry_run=True) == run_all.EXIT_OK
    archive = env / "artifacts_phase2_dry.tar.zst"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert va.verify(archive, digest) == []
    assert "archive sha256" in va.verify(archive, "0" * 64)[0]
    assert va.verify(archive, digest) == []


def test_a_damaged_member_is_caught_by_the_manifest(env, monkeypatch):
    import io

    import zstandard as zstd
    monkeypatch.setattr(va, "MANIFEST", "runs_dry/session_manifest.json")
    run_all.run_session(cfg_file(env), dry_run=True)
    archive = env / "artifacts_phase2_dry.tar.zst"
    raw = zstd.ZstdDecompressor().stream_reader(archive.open("rb")).read()
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, tarfile.open(fileobj=out, mode="w") as dst:
        for m in src:
            data = src.extractfile(m).read() if m.isfile() else None
            if m.name.endswith("p2-hospital-f2-plain-s0/model.pt"):
                mid = len(data) // 2
                data = data[:mid] + bytes([data[mid] ^ 0xFF]) + data[mid + 1:]          # flip one byte
            if data is not None:
                m.size = len(data)
                dst.addfile(m, io.BytesIO(data))
    bad = env / "bad.tar.zst"
    bad.write_bytes(zstd.ZstdCompressor().compress(out.getvalue()))
    problems = va.verify(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
    assert any("model.pt" in p and "differs" in p for p in problems)


# ------------------------------------------------------------------ refusals and failures

def test_data_that_does_not_match_the_manifest_exits_4(env):
    (env / "data" / "p2" / "meta.parquet").write_bytes(b"corrupted")
    assert run_all.run_session(cfg_file(env), dry_run=True) == run_all.EXIT_DATA
    assert "meta.parquet does not match" in (env / "runs_dry" / "session.log").read_text()


def test_a_missing_file_and_a_missing_manifest_are_reported(env):
    (env / "data" / "p2" / "splits.parquet").unlink()
    assert any("splits.parquet is missing" in p for p in run_all.verify_data(env / "data" / "p2"))
    assert run_all.verify_data(env / "nowhere") == [f"{env / 'nowhere' / 'data_manifest.json'} is missing"]


def test_budget_refusal_exits_2_before_training_anything(env):
    code = run_all.run_session(cfg_file(env, budget=0.000001), device="cpu")
    assert code == run_all.EXIT_BUDGET
    log = (env / "runs" / "session.log").read_text()
    assert "refused: projected time exceeds 90% of the budget" in log
    assert not list((env / "runs").glob("p2-*"))


def test_allow_partial_lets_the_watchdog_cut_the_queue_and_still_packs(env):
    code = run_all.run_session(cfg_file(env, budget=0.000001, partial="true"), device="cpu")
    assert code == run_all.EXIT_BUDGET
    assert "watchdog" in (env / "runs" / "session.log").read_text()
    assert (env / "artifacts_phase2.tar.zst").exists() and not list((env / "runs").glob("p2-*/DONE"))


def test_probe_only_prints_the_projection_and_trains_nothing(env):
    assert run_all.run_session(cfg_file(env), probe_only=True, device="cpu") == run_all.EXIT_OK
    assert "probe:" in (env / "runs" / "session.log").read_text()
    assert not list((env / "runs").glob("p2-*"))


def test_one_failed_run_does_not_end_the_session(env, monkeypatch):
    real = run_all.train_run

    def flaky(cfg, inputs, device, out_dir, stop):
        if run_id(cfg) == "p2-hospital-f0-plain-s0":
            raise RuntimeError("synthetic failure")
        return real(cfg, inputs, device, out_dir, stop)

    monkeypatch.setattr(run_all, "train_run", flaky)
    assert run_all.run_session(cfg_file(env), dry_run=True) == run_all.EXIT_RUN_FAILED
    runs = env / "runs_dry"
    assert "synthetic failure" in (runs / "p2-hospital-f0-plain-s0" / "error.txt").read_text()
    assert (runs / "p2-hospital-f2-aug-s0" / "DONE").exists()               # later runs went on
    assert json.loads((runs / "session_manifest.json").read_text())["failed_runs"] == ["p2-hospital-f0-plain-s0"]


def test_a_stop_signal_exits_3_and_the_same_command_resumes(env):
    cfg = cfg_file(env, epochs=2)
    assert run_all.run_session(cfg, device="cpu", stop_flag=lambda: True) == run_all.EXIT_SIGNAL
    first = env / "runs" / "p2-hospital-f2-plain-s0"
    assert (first / "ckpt.pt").exists() and not (first / "DONE").exists()
    assert run_all.run_session(cfg, device="cpu") == run_all.EXIT_OK
    assert (first / "DONE").exists() and not (first / "ckpt.pt").exists()


def test_a_gpu_with_too_little_free_memory_is_refused(env, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (8 * 2**30, 16 * 2**30))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *_: True)
    assert run_all.run_session(cfg_file(env), device="cuda") == run_all.EXIT_BUDGET
    log = (env / "runs" / "session.log").read_text()
    assert "Fake GPU" in log and "below min_gpu_gb = 20.0" in log


# ------------------------------------------------------------------ pieces

def test_sample_rows_cells_are_seeded_and_disjoint(env):
    meta = pd.read_parquet(env / "data" / "p2" / "meta.parquet")
    a = run_all.sample_rows(meta, 4, seed=1)
    assert len(a) == 40 and (np.sort(a) == a).all() and (run_all.sample_rows(meta, 4, seed=1) == a).all()
    sub = meta.loc[meta["row"].isin(a)]
    assert (sub.groupby(["hospital", "label"]).size() == 4).all()
    b = run_all.sample_rows(meta, 4, seed=2, exclude=a)
    assert not set(a) & set(b) and len(b) == 40


def test_dry_run_config_is_one_small_epoch_in_fp32():
    r = expand_queue(ROOT / "configs" / "phase2_gpu.yaml", 4)[0]
    d = run_all.dry_run_cfg(r)
    assert d.optim.epochs == 1 and d.optim.batch_size == 32 and d.compile is False and d.amp_dtype == "fp32"
    assert d.optim.peak_lr == r.optim.peak_lr and d.model == r.model


# ------------------------------------------------------------------ images that do not fit on the GPU

def test_out_of_memory_is_not_mistaken_for_a_compiler_failure():
    from pathlens import train

    class Boom:
        compiled = True
        disabled = False

        def __call__(self, x):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

        def disable(self, why):
            self.disabled = True

    st = Boom()
    with pytest.raises(torch.OutOfMemoryError):
        train._train_step(st, torch.nn.CrossEntropyLoss(), None, torch.zeros(2, 3, 4, 4), torch.zeros(2, dtype=torch.long),
                          False, torch.device("cpu"), first=True)
    assert st.disabled is False                                 # no pointless fall back to eager


def test_a_failed_compile_falls_back_to_eager_and_the_step_still_runs():
    from pathlens import train

    model = torch.nn.Linear(4, 2)

    class Flaky:
        compiled = True

        def __init__(self):
            self.fn = lambda x: (_ for _ in ()).throw(RuntimeError("inductor exploded"))

        def __call__(self, x):
            return self.fn(x)

        def disable(self, why):
            self.fn, self.compiled = model, False

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train._train_step(Flaky(), torch.nn.CrossEntropyLoss(), opt, torch.randn(3, 4), torch.tensor([0, 1, 0]),
                             False, torch.device("cpu"), first=True)
    assert torch.isfinite(loss) and model.weight.grad is not None


def test_images_that_do_not_fit_move_to_host_memory_and_the_probe_is_repeated(env, monkeypatch):
    real = run_all.measure_throughput
    calls = []

    def oom_once(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(*a, **k)

    monkeypatch.setattr(run_all, "measure_throughput", oom_once)
    assert run_all.run_session(cfg_file(env), dry_run=True) == run_all.EXIT_OK
    log = (env / "runs_dry" / "session.log").read_text()
    assert "moving them to host memory and probing again" in log and log.count("images on cpu (") == 2
    assert json.loads((env / "runs_dry" / "session_manifest.json").read_text())["probe"]["images_on"] == "host"


def test_images_on_device_is_not_second_guessed(env, monkeypatch):
    p = cfg_file(env)
    p.write_text(p.read_text() + "images_on: device\n")
    monkeypatch.setattr(run_all, "measure_throughput", lambda *a, **k: (_ for _ in ()).throw(torch.OutOfMemoryError("x")))
    with pytest.raises(torch.OutOfMemoryError):
        run_all.run_session(p, dry_run=True)


def test_gather_batch_and_predict_work_with_images_in_either_place():
    from pathlens.train import gather_batch
    imgs = torch.randint(0, 255, (10, 4, 4, 3), dtype=torch.uint8)
    idx = torch.tensor([7, 2, 2])
    assert torch.equal(gather_batch(imgs, idx, "cpu"), imgs[idx]) and gather_batch(imgs, idx, "cpu").dtype == torch.uint8
