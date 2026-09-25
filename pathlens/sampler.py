"""Epoch sampler that works on any device (architecture doc, "Access classes", DeviceSampler)."""
from __future__ import annotations

from collections.abc import Iterator

import torch


class DeviceSampler:
    """randperm of `rows` drawn on the rows' device with `generator`, then sliced into batches."""

    def __init__(self, rows: torch.Tensor, batch_size: int, generator: torch.Generator, drop_last: bool
                 ) -> None:
        if rows.dtype != torch.long or rows.ndim != 1:
            raise ValueError("rows must be a 1-D int64 tensor")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.rows, self.batch_size, self.generator, self.drop_last = rows, batch_size, generator, drop_last

    def __len__(self) -> int:
        n = len(self.rows)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def epoch(self) -> Iterator[torch.Tensor]:
        perm = torch.randperm(len(self.rows), generator=self.generator, device=self.rows.device)
        order = self.rows[perm]
        for i in range(len(self)):
            yield order[i * self.batch_size:(i + 1) * self.batch_size]
