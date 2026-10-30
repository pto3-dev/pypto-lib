# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure CPU layout proof for <=4 verifier requests, each1..6 query tokens.

Input metadata must already be a host list. This module never reads a device
Tensor, generates text, or infers lengths from MTP acceptance statistics.
"""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class Layout:
    query_lengths: tuple[int, ...]
    input_rows: int
    gather_input: tuple[int, ...]
    valid_rows: tuple[bool, ...]
    restore_output: tuple[int, ...]
    ends: tuple[int, ...]

    @property
    def canonical(self):
        return self.query_lengths == (6, 6, 6, 6) and self.input_rows == 24


def layout(query_ends, input_rows):
    if not isinstance(query_ends, (list, tuple)) or not 1 <= len(query_ends) <= 4:
        raise ValueError("Expected1..4 host cumulative query lengths")
    if any(not isinstance(x, Integral) or isinstance(x, bool) for x in query_ends):
        raise TypeError("Only host integer metadata is accepted")
    lengths = []
    previous = 0
    for end in query_ends:
        n = int(end) - previous
        if not 1 <= n <= 6:
            raise ValueError("Each verifier query must contain1..6 rows")
        lengths.append(n)
        previous = int(end)
    if not previous <= input_rows <= 24:
        raise ValueError("Framework row capacity is inconsistent with query lengths")
    gather = []
    valid = []
    restore = []
    source = 0
    for req in range(4):
        n = lengths[req] if req < len(lengths) else 0
        for local in range(6):
            gather.append(source + min(local, n - 1) if n else 0)
            valid.append(local < n)
            if local < n:
                restore.append(req * 6 + local)
        source += n
    # Framework padding remains after all real packed rows. Values are ignored
    # by the framework, but the return shape still matches its input capacity.
    restore.extend(i for i, v in enumerate(valid) if not v)
    ends = tuple(req * 6 + (lengths[req] if req < len(lengths) else 0) for req in range(4))
    return Layout(tuple(lengths), input_rows, tuple(gather), tuple(valid), tuple(restore[:input_rows]), ends)
