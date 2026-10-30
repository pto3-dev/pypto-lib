# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Paged attention math using DSL pipelines and a declared L2 bypass policy.

The tile sizes match the measured target. The device's physical core count
and cache alias distance are obtained through runtime interfaces.
"""

import pypto.language as pl


@pl.jit
def compute_scores(
    q: pl.Tensor[[24, 6, 256], pl.BF16],
    cache: pl.Tensor[[294912, 1, 256], pl.BF16],
    table: pl.Tensor[[24, 336], pl.INT32],
    lengths: pl.Tensor[[24], pl.INT32],
    out: pl.Out[pl.Tensor[[144, 43008], pl.FP32]],
) -> pl.Tensor[[144, 43008], pl.FP32]:
    tokens = 24
    qf = pl.reshape(q, [tokens * 6, 256])
    cf = pl.reshape(cache, [294912, 1 * 256])
    jobs = tokens // 6 * 1 * 28
    available = pl.system.available_cluster_count()
    width = pl.min(available, jobs)
    for core in pl.spmd(width, name_hint="s2_scores", allow_early_resolve=True):
        pl.set_cache_policy(cf, pl.CachePolicy.BYPASS)
        token = core % (tokens // 6)
        local_core = core // (tokens // 6)
        token_width = (pl.tile.get_block_num() + tokens // 6 - 1 - token) // (tokens // 6)
        visible = pl.read(lengths, [token * 6 + 5])
        for head in pl.range(1):
            qrow = token * 6 * 6 + head * 6 * 6
            qt = pl.slice(qf, [48, 256], [qrow, 0], [6 * 6, 256], pad_value=0, clamp=True)
            qt = pl.tensor.set_validshape(qt, 48, 256)
            for chunk in pl.range(local_core, 28, token_width):
                for local_offset in pl.pipeline(0, 1536, 128, stage=2):
                    offset = chunk * 1536 + local_offset
                    physical = pl.read(table, [token * 6, offset // 128]) * 128 + offset % 128
                    valid = pl.max(0, pl.min(128, visible - offset))
                    kt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                    kt = pl.tensor.set_validshape(kt, 128, 256)
                    product = pl.matmul(qt, kt, b_trans=True, out_dtype=pl.FP32)
                    out = pl.assemble(out, pl.tensor.set_validshape(product, 6 * 6, 128), [qrow, offset])
    return out


@pl.jit
def compute_partial_values(
    probability: pl.Tensor[[144, 43008], pl.BF16],
    cache: pl.Tensor[[294912, 1, 256], pl.BF16],
    table: pl.Tensor[[24, 336], pl.INT32],
    lengths: pl.Tensor[[24], pl.INT32],
    out: pl.Out[pl.Tensor[[4032, 256], pl.FP32]],
) -> pl.Tensor[[4032, 256], pl.FP32]:
    tokens = 24
    cf = pl.reshape(cache, [294912, 1 * 256])
    jobs = tokens // 6 * 1 * 28
    available = pl.system.available_cluster_count()
    width = pl.min(available, jobs)
    for core in pl.spmd(width, name_hint="s2_partial_values", allow_early_resolve=True):
        pl.set_cache_policy(cf, pl.CachePolicy.BYPASS)
        token = core % (tokens // 6)
        local_core = core // (tokens // 6)
        token_width = (pl.tile.get_block_num() + tokens // 6 - 1 - token) // (tokens // 6)
        visible = pl.read(lengths, [token * 6 + 5])
        for job in pl.range(local_core, 1 * 28, token_width):
            head = job // 28
            chunk = job % 28
            qrow = token * 6 * 6 + head * 6 * 6
            acc = pl.create_tensor([48, 256], pl.FP32)
            for local_offset in pl.pipeline(0, 1536, 128, stage=2):
                offset = chunk * 1536 + local_offset
                physical = pl.read(table, [token * 6, offset // 128]) * 128 + offset % 128
                valid = pl.max(0, pl.min(128, visible - offset))
                pt = pl.slice(probability, [48, 128], [qrow, offset], [6 * 6, 128], pad_value=0, clamp=True)
                vt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                pt = pl.tensor.set_validshape(pt, 48, 128)
                vt = pl.tensor.set_validshape(vt, 128, 256)
                acc = pl.matmul_acc(acc, pt, vt, init_cond=local_offset == 0)
            out = pl.assemble(out, pl.tensor.set_validshape(acc, 6 * 6, 256), [chunk * tokens * 6 + qrow, 0])
    return out
