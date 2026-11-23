# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""PTO3 A1 B20 metadata and native PA-cache images; independent of the B1 decoder."""

from dataclasses import dataclass
from pathlib import Path

import torch

BATCH = 20
CONTEXT = 4096
PAGE = 128
TABLE_WIDTH = CONTEXT // PAGE


@dataclass(frozen=True)
class A1BatchedMetadata:
    """Stable-address carriers refreshed before graph replay, without host scalar reads."""

    positions: torch.Tensor
    slots: torch.Tensor
    lengths: torch.Tensor
    block_table: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device | str) -> "A1BatchedMetadata":
        def zeros(width: int) -> torch.Tensor:
            return torch.zeros((BATCH, width), dtype=torch.int32, device=device)

        positions = zeros(8)
        slots = torch.full((BATCH, 8), -1, dtype=torch.int32, device=device)
        lengths = zeros(8)
        block_table = zeros(TABLE_WIDTH)
        return cls(positions, slots, lengths, block_table)

    def update(self, positions: torch.Tensor, slots: torch.Tensor, lengths: torch.Tensor,
               block_table: torch.Tensor, *, foreach_copy: bool = False) -> None:
        """Copy compact vLLM decode metadata, then clear inactive graph rows.

        Value bounds are checked by the device kernels. Input shape and dtype
        validation happens before any mutation, including failed shape updates.
        """
        if positions.ndim != 1 or not 1 <= positions.numel() <= BATCH:
            raise ValueError("A1 B20 requires 1..20 pure-decode positions")
        rows = positions.numel()
        if slots.shape != (rows,) or lengths.shape != (rows,):
            raise ValueError("A1 B20 requires one slot and length per position")
        if block_table.ndim != 2 or block_table.shape[0] != rows or not 1 <= block_table.shape[1] <= TABLE_WIDTH:
            raise ValueError("A1 B20 requires a per-request page table with 1..32 columns")
        if any(t.dtype not in (torch.int32, torch.int64) for t in (positions, slots, lengths, block_table)):
            raise ValueError("A1 B20 metadata must use INT32 or INT64")
        self.positions.zero_()
        self.slots.fill_(-1)
        self.lengths.zero_()
        self.block_table.zero_()
        if foreach_copy:
            destinations = [self.positions[:rows, 0], self.slots[:rows, 0], self.lengths[:rows, 0],
                            self.block_table[:rows, :block_table.shape[1]]]
            torch._foreach_copy_(destinations, [positions, slots, lengths, block_table])
        else:
            self.positions[:rows, 0].copy_(positions)
            self.slots[:rows, 0].copy_(slots)
            self.lengths[:rows, 0].copy_(lengths)
            self.block_table[:rows, :block_table.shape[1]].copy_(block_table)


def compile_a1_batched_cache_images(pto_isa_root: str | Path) -> dict[str, bytes]:
    """Compile bit-preserving FP16/BF16 cache copies for native [pages,32,128,16] PA layout."""
    from simpler_setup import KernelCompiler

    compiler = KernelCompiler("a1")
    root = Path(__file__).parent
    return {
        name: compiler.compile_single_launch(str(root / f"a1_kv_{name}_batched.cpp"), str(pto_isa_root))
        for name in ("append", "gather")
    }
