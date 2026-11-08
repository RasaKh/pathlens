#!/usr/bin/env python3
"""Phase 3 extractor: frozen Phikon (owkin/phikon) features for the Phase 2 subset patches.

    python scripts/extract_p3.py --data DIR --out DIR [--limit 500] [--batch 256]

Runs on the rented GPU box in its own environment (torch + transformers + numpy). It deliberately does NOT import
`pathlens`, so it can be copied alone. Reads DIR/subset_u8.npy [n,96,96,3] uint8 and DIR/subset_rows.npy; writes
OUT/p3_phikon.npy (float16 [n,768], same order as subset_rows.npy) and OUT/p3_manifest.json.

Resumable: features are written to a memmap in chunks and OUT/p3_progress.json records how many rows are done;
a stopped run continues from there. Feature = the CLS token of the last hidden state (what the Phikon model card uses).
Preprocessing: bilinear resize 96 -> 224 (the model's image processor uses PIL resample=2) and ImageNet mean/std normalisation (the model's own image processor config).
The 96 px patches being upscaled is a known confound and is stated on the page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = "owkin/phikon"
FEAT_DIM = 768
INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def extract(images, encode, out_path: Path, progress_path: Path, batch: int = 256, limit: int | None = None,
            log=print) -> int:
    """Encode `images` (uint8 [n,h,w,3], any array-like supporting slicing) with `encode(uint8 batch) -> float [b,768]`
    into a float16 memmap at `out_path`, in order, resuming from `progress_path`. Returns rows done."""
    n = len(images) if limit is None else min(limit, len(images))
    done = 0
    if progress_path.exists() and out_path.exists():
        p = json.loads(progress_path.read_text())
        if p.get("n") == n and p.get("dim") == FEAT_DIM:
            done = int(p["done"])
    mode = "r+" if (done > 0 and out_path.exists()) else "w+"
    feats = np.lib.format.open_memmap(out_path, mode=mode, dtype=np.float16, shape=(n, FEAT_DIM))
    t0 = time.time()
    for s in range(done, n, batch):
        e = min(s + batch, n)
        f = np.asarray(encode(np.ascontiguousarray(images[s:e])))
        if f.shape != (e - s, FEAT_DIM):
            raise ValueError(f"encoder returned {f.shape}, expected {(e - s, FEAT_DIM)}")
        feats[s:e] = f.astype(np.float16)
        feats.flush()
        progress_path.write_text(json.dumps({"n": n, "dim": FEAT_DIM, "done": e}))
        if ((s - done) // batch) % 20 == 0 or e == n:
            rate = (e - done) / max(time.time() - t0, 1e-9)
            log(f"{e:,}/{n:,} rows  {rate:,.0f} patches/s")
    del feats
    return n


def _revision():
    try:
        from huggingface_hub import model_info
        return model_info(MODEL_ID).sha
    except Exception:  # noqa: BLE001  (offline or API error: the revision is recorded as unknown)
        return None


def make_encoder(device: str = "cuda"):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import ViTModel

    model = ViTModel.from_pretrained(MODEL_ID, add_pooling_layer=False).to(device).eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    amp = device == "cuda"

    @torch.inference_mode()
    def encode(batch_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_u8).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False).clamp_(0, 1)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(pixel_values=x).last_hidden_state[:, 0, :]
        return out.float().cpu().numpy()

    info = {"transformers": transformers.__version__, "torch": torch.__version__,
            "revision": _revision(), "autocast_fp16": amp}
    return encode, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with subset_u8.npy and subset_rows.npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (speed and sanity check)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    out, prog = a.out / f"p3_phikon{suffix}.npy", a.out / f"p3_progress{suffix}.json"
    images = np.load(a.data / "subset_u8.npy", mmap_mode="r")
    rows = np.load(a.data / "subset_rows.npy")
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[3] != 3 or len(images) != len(rows):
        sys.exit(f"unexpected subset: {images.dtype} {images.shape}, rows {rows.shape}")
    encode, info = make_encoder(a.device)
    t0 = time.time()
    n = extract(images, encode, out, prog, batch=a.batch, limit=a.limit)
    wall = time.time() - t0
    f = np.load(out, mmap_mode="r")
    if f.shape != (n, FEAT_DIM) or f.dtype != np.float16:
        sys.exit(f"bad output {f.shape} {f.dtype}")
    if not np.isfinite(np.asarray(f[: min(n, 20000)], dtype=np.float32)).all():
        sys.exit("non-finite features")
    manifest = {"model_id": MODEL_ID, "feature": "CLS token of last_hidden_state", "input_size": INPUT_SIZE,
                "resize": "bilinear 96->224", "mean": MEAN, "std": STD, "n_patches": int(n),
                "limit": a.limit, "dtype": "float16", "dim": FEAT_DIM, **info,
                "sha256_output": sha256_file(out), "sha256_subset": sha256_file(a.data / "subset_u8.npy"),
                "wall_seconds_this_run": round(wall, 1)}
    (a.out / f"p3_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
