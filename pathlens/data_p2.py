"""Phase 2 data access (architecture doc, "Access classes"): Camelyon17-WILDS patches in one memmap.

`P2Store(data_dir)` takes the data folder (`$PATHLENS_HOME/data`) and reads `data_dir/p2/`. The CUDA path of
`to_device` (pinned host buffer, chunked copy) has not been run on a GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

N_PATCHES = 455_954
PATCH = 96


def patch_relpath(patient: str, node: int, x: int, y: int) -> str:
    """Where WILDS keeps a patch (same string as Camelyon17Dataset builds); `patient` keeps its leading zeros."""
    return f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"


def decode_rows(args: tuple[str, list[str], int]) -> int:
    """Pool worker: decode PNGs and write them into the shared memmap in place. Raises on a wrong shape."""
    npy_path, paths, start = args
    mm = np.load(npy_path, mmap_mode="r+")
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB"))
        if img.shape != (PATCH, PATCH, 3) or img.dtype != np.uint8:
            raise ValueError(f"{p}: expected {PATCH}x{PATCH}x3 uint8, got {img.shape} {img.dtype}")
        mm[start + k] = img
    mm.flush()
    return len(paths)


class P2Store:
    def __init__(self, data_dir: Path, mmap: bool = True) -> None:
        import pandas as pd

        root = Path(data_dir) / "p2"
        self.meta = pd.read_parquet(root / "meta.parquet")
        self.patches = np.load(root / "patches_u8.npy", mmap_mode="r" if mmap else None)
        if len(self.meta) != len(self.patches):
            raise ValueError(f"meta has {len(self.meta)} rows but patches_u8.npy has {len(self.patches)}")

    def rows_where(self, **eq) -> np.ndarray:
        """Row indices (int64, ascending) where every column equals the given value, e.g. hospital=2."""
        mask = np.ones(len(self.meta), dtype=bool)
        for col, val in eq.items():
            if col not in self.meta.columns:
                raise KeyError(f"no column {col!r}")
            mask &= (self.meta[col] == val).to_numpy()
        return np.flatnonzero(mask)

    def gather_cpu(self, rows: np.ndarray) -> torch.Tensor:
        """uint8 [B,96,96,3] for `rows`, in the order asked. The memmap is read in sorted order (locality)
        and put back into the requested order."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        out = np.empty((len(rows), PATCH, PATCH, 3), dtype=np.uint8)
        out[order] = self.patches[rows[order]]
        return torch.from_numpy(out)

    def to_device(self, device, chunk_bytes: int = 1 << 30) -> torch.Tensor:
        """The whole array on `device`, copied in chunks of about `chunk_bytes` through a pinned buffer so host
        RAM never holds a second full copy."""
        device = torch.device(device)
        n = len(self.patches)
        out = torch.empty((n, PATCH, PATCH, 3), dtype=torch.uint8, device=device)
        step = max(1, chunk_bytes // (PATCH * PATCH * 3))
        pin = device.type == "cuda"
        buf = torch.empty((min(step, n), PATCH, PATCH, 3), dtype=torch.uint8, pin_memory=pin)
        for a in range(0, n, step):
            b = min(a + step, n)
            buf[: b - a].numpy()[:] = self.patches[a:b]
            out[a:b].copy_(buf[: b - a], non_blocking=pin)
        if pin:
            torch.cuda.synchronize(device)
        return out
