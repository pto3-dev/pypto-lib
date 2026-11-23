# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit measured native cache geometry for this compile specialization.

Set QWEN36_NATIVE_GEOMETRY to a rankN_dimensions.json produced by the frozen
geometry-only interface. This is configuration loading, never model rewriting.
No historical cache-shape fallback is permitted.
"""
import json,os
from pathlib import Path
SEQUENCE=45056
PAGE=128
CHUNK=1536
CHUNKS=(SEQUENCE+CHUNK-1)//CHUNK
PARTIAL_ROWS=144*CHUNKS
TABLE_COLS=SEQUENCE//PAGE
TABLE_WORDS=24*TABLE_COLS
METADATA_WORDS=72+TABLE_WORDS
CONTROL_WORDS=101+TABLE_WORDS
CONTROL_PAD=(CONTROL_WORDS+7)//8*8
SOFTMAX_SLOTS=SEQUENCE//1024
path=os.environ.get("QWEN36_NATIVE_GEOMETRY")
if not path:raise RuntimeError("Pass an explicit measured rank geometry via QWEN36_NATIVE_GEOMETRY")
MEASURED_GEOMETRY_PATH=Path(path).resolve()
GEOMETRY=json.loads(MEASURED_GEOMETRY_PATH.read_text())
if GEOMETRY["max_seq"]!=SEQUENCE or GEOMETRY["cache_block_size"]!=PAGE:
    raise ValueError("Measured sequence/page geometry does not match this specialization")
CACHE_ROWS=GEOMETRY["cache_rows"]
STATE_SLOTS=GEOMETRY["state_slots"]
if CACHE_ROWS<=0 or STATE_SLOTS<=0 or not GEOMETRY["native_cache_layout"]:
    raise ValueError("Incomplete measured native cache contract")
