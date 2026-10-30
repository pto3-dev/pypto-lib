# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full mixed-W8A8 target/verifier: direct source, no legacy generator.

The 24 rows are B4 x (5 drafts + 1 verifier token). Rank selection only
chooses an explicit embedding shard; operators and the graph are shared.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto import ir
from .operators.attention import (
    norm_rope_q,
    norm_rope_k_append,
    reduce_gate,
    compute_scores,
    softmax_scores,
    compute_partial_values,
)
from .operators.collectives import (
    reduce_embedding,
    gather_logits,
    reduce_add,
    reduce_add_norm,
    publish_hidden,
    graph_stop_fence,
    reduce_add_norm_quant,
)
from .operators.embedding import embedding0, embedding1, embedding2, embedding3
from .operators.gdn import convolve, recurrent, gates
from .operators.norm_pointwise import norm_plain, norm_hidden, norm_gdn
from .operators.projection_bf16 import project_attention_out, project_gdn, project_z, project_head
from .operators.quantized import mlp_from_quant, project_down, project_qkv, project_attention_out_w8a8
from .operators.registry import OPERATORS


def build_program(rank, variant="reference"):
    if rank not in range(4):
        raise ValueError("TP4 rank must be 0..3")
    down_projection = project_down
    norm_quant_reduction = reduce_add_norm_quant
    norm_reduction = reduce_add_norm
    add_reduction = reduce_add
    operators = OPERATORS
    if variant == "down_l1k2176":
        from .operators import down_prefetch

        down_projection = down_prefetch.project_down
        operators = [f for f in OPERATORS if f.name not in ("project_down", "project_down_kernel")] + [
            down_prefetch.project_down_kernel,
            down_prefetch.project_down,
        ]
    elif variant in ("comm_lanes", "comm_all_lanes"):
        from .operators import collective_lanes

        norm_quant_reduction = collective_lanes.reduce_add_norm_quant
        operators = [
            f for f in OPERATORS if f.name not in ("reduce_add_norm_quant", "reduce_add_norm_quant_kernel")
        ] + [collective_lanes.reduce_add_norm_quant_kernel, collective_lanes.reduce_add_norm_quant]
        if variant == "comm_all_lanes":
            from .operators import collective_even_lanes

            norm_reduction = collective_even_lanes.reduce_add_norm
            add_reduction = collective_even_lanes.reduce_add
            operators = [
                f
                for f in operators
                if f.name
                not in ("reduce_add_norm", "reduce_add_norm_kernel", "reduce_add", "reduce_add_kernel")
            ] + [
                collective_even_lanes.reduce_add_norm_kernel,
                collective_even_lanes.reduce_add_norm,
                collective_even_lanes.reduce_add_kernel,
                collective_even_lanes.reduce_add,
            ]
    elif variant != "reference":
        raise ValueError("Unknown explicit tuning variant")
    selected_embedding = (embedding0, embedding1, embedding2, embedding3)[rank]

    @pl.function(type=pl.FunctionType.Orchestration)
    def target(
        control_i64: pl.Tensor[[72], pl.INT64],
        control_i32: pl.InOut[pl.Tensor[[8165], pl.INT32]],
        native_metadata: pl.Tensor[[64, 8136], pl.INT32],
        embedding: pl.Tensor[[62080, 5120], pl.BF16],
        final_norm: pl.Tensor[[1, 5120], pl.BF16],
        lm_head: pl.Tensor[[62080, 5120], pl.BF16, pl.NZ],
        weights0: pl.Tensor[[441216], pl.BF16],
        weights1: pl.Tensor[[441216], pl.BF16],
        weights2: pl.Tensor[[441216], pl.BF16],
        weights3: pl.Tensor[[441216], pl.BF16],
        weights4: pl.Tensor[[441216], pl.BF16],
        weights5: pl.Tensor[[441216], pl.BF16],
        weights6: pl.Tensor[[441216], pl.BF16],
        weights7: pl.Tensor[[441216], pl.BF16],
        weights8: pl.Tensor[[441216], pl.BF16],
        weights9: pl.Tensor[[441216], pl.BF16],
        weights10: pl.Tensor[[441216], pl.BF16],
        weights11: pl.Tensor[[441216], pl.BF16],
        weights12: pl.Tensor[[441216], pl.BF16],
        weights13: pl.Tensor[[441216], pl.BF16],
        weights14: pl.Tensor[[441216], pl.BF16],
        weights15: pl.Tensor[[441216], pl.BF16],
        float_weights: pl.Tensor[[1152], pl.FP32],
        scratch_hidden: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_residual: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_norm: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_attention: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_mixed: pl.InOut[pl.Tensor[[24, 2560], pl.BF16]],
        scratch_z: pl.InOut[pl.Tensor[[24, 1536], pl.BF16]],
        scratch_a: pl.InOut[pl.Tensor[[24, 12], pl.BF16]],
        scratch_b: pl.InOut[pl.Tensor[[24, 12], pl.BF16]],
        scratch_convolved: pl.InOut[pl.Tensor[[24, 2560], pl.BF16]],
        scratch_q: pl.InOut[pl.Tensor[[24, 4, 128], pl.BF16]],
        scratch_k: pl.InOut[pl.Tensor[[24, 4, 128], pl.BF16]],
        scratch_v: pl.InOut[pl.Tensor[[24, 12, 128], pl.BF16]],
        scratch_gdn_core: pl.InOut[pl.Tensor[[24, 12, 128], pl.BF16]],
        scratch_gdn_normal: pl.InOut[pl.Tensor[[288, 128], pl.BF16]],
        scratch_packed_q: pl.InOut[pl.Tensor[[24, 3072], pl.BF16]],
        scratch_full_q: pl.InOut[pl.Tensor[[24, 6, 256], pl.BF16]],
        scratch_full_gate: pl.InOut[pl.Tensor[[24, 1536], pl.BF16]],
        scratch_full_k: pl.InOut[pl.Tensor[[24, 256], pl.BF16]],
        scratch_full_v: pl.InOut[pl.Tensor[[24, 256], pl.BF16]],
        scratch_full_qn: pl.InOut[pl.Tensor[[144, 256], pl.BF16]],
        scratch_full_kn: pl.InOut[pl.Tensor[[24, 256], pl.BF16]],
        scratch_full_qr: pl.InOut[pl.Tensor[[24, 6, 256], pl.BF16]],
        scratch_full_kr: pl.InOut[pl.Tensor[[24, 1, 256], pl.BF16]],
        scratch_full_core: pl.InOut[pl.Tensor[[24, 6, 256], pl.BF16]],
        scratch_gated: pl.InOut[pl.Tensor[[24, 1536], pl.BF16]],
        scratch_mlp_norm: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_mlp_gate: pl.InOut[pl.Tensor[[24, 4352], pl.BF16]],
        scratch_mlp_up: pl.InOut[pl.Tensor[[24, 4352], pl.BF16]],
        scratch_mlp_act: pl.InOut[pl.Tensor[[24, 4352], pl.BF16]],
        scratch_mlp_down: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_head_norm: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        scratch_logits: pl.InOut[pl.Tensor[[24, 62080], pl.BF16]],
        scratch_mlp_packed: pl.InOut[pl.Tensor[[24, 8704], pl.BF16]],
        scratch_attention_probability: pl.InOut[pl.Tensor[[144, 43008], pl.BF16]],
        float_qn: pl.InOut[pl.Tensor[[24, 4, 128], pl.FP32]],
        float_kn: pl.InOut[pl.Tensor[[24, 4, 128], pl.FP32]],
        float_alpha: pl.InOut[pl.Tensor[[24, 12], pl.FP32]],
        float_beta: pl.InOut[pl.Tensor[[24, 12], pl.FP32]],
        float_partial: pl.InOut[pl.Tensor[[24, 5120], pl.FP32]],
        float_attention_scores: pl.InOut[pl.Tensor[[144, 43008], pl.FP32]],
        float_attention_partials: pl.InOut[pl.Tensor[[4032, 256], pl.FP32]],
        l0_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l0_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l1_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l1_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l2_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l2_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l3_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l3_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l4_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l4_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l5_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l5_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l6_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l6_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l7_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l7_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l8_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l8_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l9_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l9_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l10_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l10_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l11_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l11_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l12_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l12_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l13_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l13_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l14_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l14_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l15_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l15_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l16_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l16_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l17_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l17_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l18_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l18_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l19_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l19_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l20_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l20_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l21_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l21_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l22_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l22_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l23_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l23_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l24_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l24_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l25_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l25_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l26_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l26_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l27_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l27_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l28_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l28_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l29_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l29_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l30_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l30_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l31_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l31_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l32_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l32_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l33_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l33_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l34_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l34_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l35_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l35_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l36_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l36_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l37_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l37_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l38_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l38_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l39_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l39_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l40_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l40_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l41_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l41_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l42_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l42_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l43_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l43_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l44_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l44_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l45_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l45_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l46_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l46_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l47_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l47_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l48_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l48_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l49_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l49_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l50_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l50_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l51_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l51_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l52_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l52_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l53_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l53_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l54_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l54_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l55_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l55_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l56_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l56_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l57_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l57_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l58_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l58_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l59_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l59_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l60_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l60_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l61_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l61_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l62_conv_state: pl.InOut[pl.Tensor[[192, 8, 2560], pl.BF16]],
        l62_recurrent_state: pl.InOut[pl.Tensor[[192, 12, 128, 128], pl.FP32]],
        l63_keys: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        l63_values: pl.InOut[pl.Tensor[[294912, 1, 256], pl.BF16]],
        cos: pl.Tensor[[43008, 32], pl.BF16],
        sin: pl.Tensor[[43008, 32], pl.BF16],
        out: pl.Out[pl.Tensor[[24, 62080], pl.BF16]],
        global_logits: pl.InOut[pl.Tensor[[24, 248320], pl.FP32]],
        hidden_result: pl.InOut[pl.Tensor[[24, 5120], pl.BF16]],
        softmax_columns: pl.Tensor[[1, 1024], pl.FP32],
        device_program_storage: pl.InOut[pl.Tensor[[13645824], pl.UINT8]],
        logit_data: pl.InOut[pld.DistributedTensor[[24, 62080], pl.BF16]],
        logit_signal: pl.InOut[pld.DistributedTensor[[768, 32], pl.INT32]],
        data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
        bank0_gateup: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
        bank0_down: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
        bank0_out: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ],
        bank0_qkvz: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
        bank1_gateup: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
        bank1_down: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
        bank1_out: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ],
        bank1_qkvz: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
        bank2_gateup: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
        bank2_down: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
        bank2_out: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ],
        bank2_qkvz: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
        bank3_gateup: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
        bank3_down: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
        bank3_out: pl.Tensor[[16, 1536, 5120], pl.INT8, pl.NZ],
        bank3_qkv: pl.Tensor[[16, 5120, 3584], pl.INT8, pl.NZ],
        bank0_gateup_scale: pl.Tensor[[16, 8704], pl.FP32],
        bank0_down_scale: pl.Tensor[[16, 5120], pl.FP32],
        bank1_gateup_scale: pl.Tensor[[16, 8704], pl.FP32],
        bank1_down_scale: pl.Tensor[[16, 5120], pl.FP32],
        bank2_gateup_scale: pl.Tensor[[16, 8704], pl.FP32],
        bank2_down_scale: pl.Tensor[[16, 5120], pl.FP32],
        bank3_gateup_scale: pl.Tensor[[16, 8704], pl.FP32],
        bank3_down_scale: pl.Tensor[[16, 5120], pl.FP32],
        bank3_out_scale: pl.Tensor[[16, 5120], pl.FP32],
        bank3_qkv_scale: pl.Tensor[[16, 3584], pl.FP32],
    ) -> pl.Tensor[[24, 248320], pl.FP32]:
        if pl.read(control_i32, [3]) == 0:
            ids = pl.reshape(pl.slice(control_i64, [24], [0]), [24])
            positions = pl.reshape(pl.slice(control_i64, [24], [24]), [24])
            kv_slots = pl.reshape(pl.slice(control_i64, [24], [48]), [24])
            metadata = pl.reshape(pl.slice(control_i32, [4], [0]), [4])
            starts = pl.reshape(pl.slice(control_i32, [25], [4]), [25])
            hidden = scratch_hidden
            residual = scratch_residual
            norm = scratch_norm
            attention = scratch_attention
            mixed = scratch_mixed
            z = scratch_z
            a = scratch_a
            b = scratch_b
            convolved = scratch_convolved
            q = scratch_q
            k = scratch_k
            v = scratch_v
            gdn_core = scratch_gdn_core
            gdn_normal = scratch_gdn_normal
            packed_q = scratch_packed_q
            full_q = scratch_full_q
            full_gate = scratch_full_gate
            full_k = scratch_full_k
            full_v = scratch_full_v
            full_qn = scratch_full_qn
            full_kn = scratch_full_kn
            full_qr = scratch_full_qr
            full_kr = scratch_full_kr
            full_core = scratch_full_core
            gated = scratch_gated
            mlp_norm = scratch_mlp_norm
            mlp_gate = scratch_mlp_gate
            mlp_up = scratch_mlp_up
            mlp_act = scratch_mlp_act
            mlp_down = scratch_mlp_down
            head_norm = scratch_head_norm
            logits = scratch_logits
            mlp_packed = scratch_mlp_packed
            attention_probability = scratch_attention_probability
            qn = float_qn
            kn = float_kn
            alpha = float_alpha
            beta = float_beta
            partial = float_partial
            attention_scores = float_attention_scores
            attention_partials = float_attention_partials
            hidden = selected_embedding(ids, embedding, hidden)
            hidden, data, signal = reduce_embedding(hidden, hidden, data, signal, metadata, 0)
            mlp_quantized = pl.create_tensor([24, 5120], pl.INT8)
            mlp_activation_scale = pl.create_tensor([24, 1], pl.FP32)
            for group_index in pl.range(16):
                if group_index == 0:
                    group_weights = weights0
                    conv0 = l0_conv_state
                    state0 = l0_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [0, 0]), [8136])
                    conv1 = l1_conv_state
                    state1 = l1_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [1, 0]), [8136])
                    conv2 = l2_conv_state
                    state2 = l2_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [2, 0]), [8136])
                    keys3 = l3_keys
                    values3 = l3_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [3, 0]), [8136])
                elif group_index == 1:
                    group_weights = weights1
                    conv0 = l4_conv_state
                    state0 = l4_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [4, 0]), [8136])
                    conv1 = l5_conv_state
                    state1 = l5_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [5, 0]), [8136])
                    conv2 = l6_conv_state
                    state2 = l6_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [6, 0]), [8136])
                    keys3 = l7_keys
                    values3 = l7_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [7, 0]), [8136])
                elif group_index == 2:
                    group_weights = weights2
                    conv0 = l8_conv_state
                    state0 = l8_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [8, 0]), [8136])
                    conv1 = l9_conv_state
                    state1 = l9_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [9, 0]), [8136])
                    conv2 = l10_conv_state
                    state2 = l10_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [10, 0]), [8136])
                    keys3 = l11_keys
                    values3 = l11_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [11, 0]), [8136])
                elif group_index == 3:
                    group_weights = weights3
                    conv0 = l12_conv_state
                    state0 = l12_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [12, 0]), [8136])
                    conv1 = l13_conv_state
                    state1 = l13_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [13, 0]), [8136])
                    conv2 = l14_conv_state
                    state2 = l14_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [14, 0]), [8136])
                    keys3 = l15_keys
                    values3 = l15_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [15, 0]), [8136])
                elif group_index == 4:
                    group_weights = weights4
                    conv0 = l16_conv_state
                    state0 = l16_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [16, 0]), [8136])
                    conv1 = l17_conv_state
                    state1 = l17_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [17, 0]), [8136])
                    conv2 = l18_conv_state
                    state2 = l18_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [18, 0]), [8136])
                    keys3 = l19_keys
                    values3 = l19_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [19, 0]), [8136])
                elif group_index == 5:
                    group_weights = weights5
                    conv0 = l20_conv_state
                    state0 = l20_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [20, 0]), [8136])
                    conv1 = l21_conv_state
                    state1 = l21_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [21, 0]), [8136])
                    conv2 = l22_conv_state
                    state2 = l22_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [22, 0]), [8136])
                    keys3 = l23_keys
                    values3 = l23_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [23, 0]), [8136])
                elif group_index == 6:
                    group_weights = weights6
                    conv0 = l24_conv_state
                    state0 = l24_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [24, 0]), [8136])
                    conv1 = l25_conv_state
                    state1 = l25_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [25, 0]), [8136])
                    conv2 = l26_conv_state
                    state2 = l26_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [26, 0]), [8136])
                    keys3 = l27_keys
                    values3 = l27_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [27, 0]), [8136])
                elif group_index == 7:
                    group_weights = weights7
                    conv0 = l28_conv_state
                    state0 = l28_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [28, 0]), [8136])
                    conv1 = l29_conv_state
                    state1 = l29_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [29, 0]), [8136])
                    conv2 = l30_conv_state
                    state2 = l30_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [30, 0]), [8136])
                    keys3 = l31_keys
                    values3 = l31_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [31, 0]), [8136])
                elif group_index == 8:
                    group_weights = weights8
                    conv0 = l32_conv_state
                    state0 = l32_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [32, 0]), [8136])
                    conv1 = l33_conv_state
                    state1 = l33_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [33, 0]), [8136])
                    conv2 = l34_conv_state
                    state2 = l34_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [34, 0]), [8136])
                    keys3 = l35_keys
                    values3 = l35_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [35, 0]), [8136])
                elif group_index == 9:
                    group_weights = weights9
                    conv0 = l36_conv_state
                    state0 = l36_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [36, 0]), [8136])
                    conv1 = l37_conv_state
                    state1 = l37_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [37, 0]), [8136])
                    conv2 = l38_conv_state
                    state2 = l38_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [38, 0]), [8136])
                    keys3 = l39_keys
                    values3 = l39_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [39, 0]), [8136])
                elif group_index == 10:
                    group_weights = weights10
                    conv0 = l40_conv_state
                    state0 = l40_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [40, 0]), [8136])
                    conv1 = l41_conv_state
                    state1 = l41_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [41, 0]), [8136])
                    conv2 = l42_conv_state
                    state2 = l42_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [42, 0]), [8136])
                    keys3 = l43_keys
                    values3 = l43_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [43, 0]), [8136])
                elif group_index == 11:
                    group_weights = weights11
                    conv0 = l44_conv_state
                    state0 = l44_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [44, 0]), [8136])
                    conv1 = l45_conv_state
                    state1 = l45_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [45, 0]), [8136])
                    conv2 = l46_conv_state
                    state2 = l46_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [46, 0]), [8136])
                    keys3 = l47_keys
                    values3 = l47_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [47, 0]), [8136])
                elif group_index == 12:
                    group_weights = weights12
                    conv0 = l48_conv_state
                    state0 = l48_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [48, 0]), [8136])
                    conv1 = l49_conv_state
                    state1 = l49_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [49, 0]), [8136])
                    conv2 = l50_conv_state
                    state2 = l50_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [50, 0]), [8136])
                    keys3 = l51_keys
                    values3 = l51_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [51, 0]), [8136])
                elif group_index == 13:
                    group_weights = weights13
                    conv0 = l52_conv_state
                    state0 = l52_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [52, 0]), [8136])
                    conv1 = l53_conv_state
                    state1 = l53_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [53, 0]), [8136])
                    conv2 = l54_conv_state
                    state2 = l54_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [54, 0]), [8136])
                    keys3 = l55_keys
                    values3 = l55_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [55, 0]), [8136])
                elif group_index == 14:
                    group_weights = weights14
                    conv0 = l56_conv_state
                    state0 = l56_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [56, 0]), [8136])
                    conv1 = l57_conv_state
                    state1 = l57_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [57, 0]), [8136])
                    conv2 = l58_conv_state
                    state2 = l58_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [58, 0]), [8136])
                    keys3 = l59_keys
                    values3 = l59_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [59, 0]), [8136])
                else:
                    group_weights = weights15
                    conv0 = l60_conv_state
                    state0 = l60_recurrent_state
                    meta0 = pl.reshape(pl.slice(native_metadata, [1, 8136], [60, 0]), [8136])
                    conv1 = l61_conv_state
                    state1 = l61_recurrent_state
                    meta1 = pl.reshape(pl.slice(native_metadata, [1, 8136], [61, 0]), [8136])
                    conv2 = l62_conv_state
                    state2 = l62_recurrent_state
                    meta2 = pl.reshape(pl.slice(native_metadata, [1, 8136], [62, 0]), [8136])
                    keys3 = l63_keys
                    values3 = l63_values
                    meta3 = pl.reshape(pl.slice(native_metadata, [1, 8136], [63, 0]), [8136])
                w0_input_layernorm_weight = pl.reshape(pl.slice(group_weights, [5120], [0]), [1, 5120])
                w0_post_attention_layernorm_weight = pl.reshape(
                    pl.slice(group_weights, [5120], [5120]), [1, 5120]
                )
                w0_linear_attn_in_proj_b_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [10240]), [12, 5120]
                )
                w0_linear_attn_in_proj_a_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [71680]), [12, 5120]
                )
                w0_linear_attn_conv1d_weight = pl.reshape(
                    pl.slice(group_weights, [10240], [133120]), [4, 2560]
                )
                w0_linear_attn_norm_weight = pl.reshape(pl.slice(group_weights, [128], [143360]), [1, 128])
                indices = pl.reshape(pl.slice(meta0, [24], [0]), [24])
                initial = pl.reshape(pl.slice(meta0, [24], [24]), [24])
                lengths = pl.reshape(pl.slice(meta0, [24], [48]), [24])
                table = pl.reshape(pl.slice(meta0, [8064], [72]), [24, 336])
                if group_index == 0:
                    norm = norm_plain(hidden, w0_input_layernorm_weight, hidden, norm)
                else:
                    norm = norm_hidden(hidden, w0_input_layernorm_weight, hidden, norm)
                z = project_z(norm, bank0_qkvz, z, group_index)
                mixed = project_gdn(norm, bank0_qkvz, mixed, group_index)
                a_log0 = pl.reshape(pl.slice(float_weights, [12], [(group_index * 3 + 0) * 24]), [1, 12])
                dt_bias0 = pl.reshape(
                    pl.slice(float_weights, [12], [(group_index * 3 + 0) * 24 + 12]), [1, 12]
                )
                ba_weight0 = pl.reshape(pl.slice(group_weights, [122880], [10240]), [24, 5120])
                alpha, beta = gates(norm, ba_weight0, a_log0, dt_bias0, alpha, beta)
                convolved, qn, kn, v, conv0 = convolve(
                    mixed,
                    w0_linear_attn_conv1d_weight,
                    conv0,
                    starts,
                    indices,
                    initial,
                    lengths,
                    convolved,
                    qn,
                    kn,
                    v,
                )
                gdn_core, state0 = recurrent(
                    qn, kn, v, alpha, beta, state0, starts, indices, initial, gdn_core
                )
                argument_view_0 = pl.reshape(gdn_core, [288, 128])
                argument_view_1 = pl.reshape(z, [288, 128])
                gdn_normal = norm_gdn(
                    argument_view_0, w0_linear_attn_norm_weight, argument_view_1, gdn_normal
                )
                argument_view_2 = pl.reshape(gdn_normal, [24, 1536])
                partial = project_attention_out(argument_view_2, bank0_out, partial, group_index)
                argument_view_3 = group_index * 8 + 1
                residual, mlp_norm, data, signal, mlp_quantized, mlp_activation_scale = norm_quant_reduction(
                    partial,
                    hidden,
                    residual,
                    w0_post_attention_layernorm_weight,
                    mlp_norm,
                    data,
                    signal,
                    metadata,
                    argument_view_3,
                    mlp_quantized,
                    mlp_activation_scale,
                )
                mlp_act = mlp_from_quant(
                    mlp_quantized,
                    bank0_gateup,
                    mlp_act,
                    group_index,
                    bank0_gateup_scale,
                    mlp_activation_scale,
                )
                partial = down_projection(mlp_act, bank0_down, partial, group_index, bank0_down_scale)
                next_input_norm0 = pl.reshape(pl.slice(group_weights, [5120], [143488]), [1, 5120])
                argument_view_4 = group_index * 8 + 2
                hidden, norm, data, signal = norm_reduction(
                    partial, residual, hidden, next_input_norm0, norm, data, signal, metadata, argument_view_4
                )
                w1_input_layernorm_weight = pl.reshape(pl.slice(group_weights, [5120], [143488]), [1, 5120])
                w1_post_attention_layernorm_weight = pl.reshape(
                    pl.slice(group_weights, [5120], [148608]), [1, 5120]
                )
                w1_linear_attn_in_proj_b_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [153728]), [12, 5120]
                )
                w1_linear_attn_in_proj_a_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [215168]), [12, 5120]
                )
                w1_linear_attn_conv1d_weight = pl.reshape(
                    pl.slice(group_weights, [10240], [276608]), [4, 2560]
                )
                w1_linear_attn_norm_weight = pl.reshape(pl.slice(group_weights, [128], [286848]), [1, 128])
                indices = pl.reshape(pl.slice(meta1, [24], [0]), [24])
                initial = pl.reshape(pl.slice(meta1, [24], [24]), [24])
                lengths = pl.reshape(pl.slice(meta1, [24], [48]), [24])
                table = pl.reshape(pl.slice(meta1, [8064], [72]), [24, 336])
                z = project_z(norm, bank1_qkvz, z, group_index)
                mixed = project_gdn(norm, bank1_qkvz, mixed, group_index)
                a_log1 = pl.reshape(pl.slice(float_weights, [12], [(group_index * 3 + 1) * 24]), [1, 12])
                dt_bias1 = pl.reshape(
                    pl.slice(float_weights, [12], [(group_index * 3 + 1) * 24 + 12]), [1, 12]
                )
                ba_weight1 = pl.reshape(pl.slice(group_weights, [122880], [153728]), [24, 5120])
                alpha, beta = gates(norm, ba_weight1, a_log1, dt_bias1, alpha, beta)
                convolved, qn, kn, v, conv1 = convolve(
                    mixed,
                    w1_linear_attn_conv1d_weight,
                    conv1,
                    starts,
                    indices,
                    initial,
                    lengths,
                    convolved,
                    qn,
                    kn,
                    v,
                )
                gdn_core, state1 = recurrent(
                    qn, kn, v, alpha, beta, state1, starts, indices, initial, gdn_core
                )
                argument_view_5 = pl.reshape(gdn_core, [288, 128])
                argument_view_6 = pl.reshape(z, [288, 128])
                gdn_normal = norm_gdn(
                    argument_view_5, w1_linear_attn_norm_weight, argument_view_6, gdn_normal
                )
                argument_view_7 = pl.reshape(gdn_normal, [24, 1536])
                partial = project_attention_out(argument_view_7, bank1_out, partial, group_index)
                argument_view_8 = group_index * 8 + 3
                residual, mlp_norm, data, signal, mlp_quantized, mlp_activation_scale = norm_quant_reduction(
                    partial,
                    hidden,
                    residual,
                    w1_post_attention_layernorm_weight,
                    mlp_norm,
                    data,
                    signal,
                    metadata,
                    argument_view_8,
                    mlp_quantized,
                    mlp_activation_scale,
                )
                mlp_act = mlp_from_quant(
                    mlp_quantized,
                    bank1_gateup,
                    mlp_act,
                    group_index,
                    bank1_gateup_scale,
                    mlp_activation_scale,
                )
                partial = down_projection(mlp_act, bank1_down, partial, group_index, bank1_down_scale)
                next_input_norm1 = pl.reshape(pl.slice(group_weights, [5120], [286976]), [1, 5120])
                argument_view_9 = group_index * 8 + 4
                hidden, norm, data, signal = norm_reduction(
                    partial, residual, hidden, next_input_norm1, norm, data, signal, metadata, argument_view_9
                )
                w2_input_layernorm_weight = pl.reshape(pl.slice(group_weights, [5120], [286976]), [1, 5120])
                w2_post_attention_layernorm_weight = pl.reshape(
                    pl.slice(group_weights, [5120], [292096]), [1, 5120]
                )
                w2_linear_attn_in_proj_b_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [297216]), [12, 5120]
                )
                w2_linear_attn_in_proj_a_weight = pl.reshape(
                    pl.slice(group_weights, [61440], [358656]), [12, 5120]
                )
                w2_linear_attn_conv1d_weight = pl.reshape(
                    pl.slice(group_weights, [10240], [420096]), [4, 2560]
                )
                w2_linear_attn_norm_weight = pl.reshape(pl.slice(group_weights, [128], [430336]), [1, 128])
                indices = pl.reshape(pl.slice(meta2, [24], [0]), [24])
                initial = pl.reshape(pl.slice(meta2, [24], [24]), [24])
                lengths = pl.reshape(pl.slice(meta2, [24], [48]), [24])
                table = pl.reshape(pl.slice(meta2, [8064], [72]), [24, 336])
                z = project_z(norm, bank2_qkvz, z, group_index)
                mixed = project_gdn(norm, bank2_qkvz, mixed, group_index)
                a_log2 = pl.reshape(pl.slice(float_weights, [12], [(group_index * 3 + 2) * 24]), [1, 12])
                dt_bias2 = pl.reshape(
                    pl.slice(float_weights, [12], [(group_index * 3 + 2) * 24 + 12]), [1, 12]
                )
                ba_weight2 = pl.reshape(pl.slice(group_weights, [122880], [297216]), [24, 5120])
                alpha, beta = gates(norm, ba_weight2, a_log2, dt_bias2, alpha, beta)
                convolved, qn, kn, v, conv2 = convolve(
                    mixed,
                    w2_linear_attn_conv1d_weight,
                    conv2,
                    starts,
                    indices,
                    initial,
                    lengths,
                    convolved,
                    qn,
                    kn,
                    v,
                )
                gdn_core, state2 = recurrent(
                    qn, kn, v, alpha, beta, state2, starts, indices, initial, gdn_core
                )
                argument_view_10 = pl.reshape(gdn_core, [288, 128])
                argument_view_11 = pl.reshape(z, [288, 128])
                gdn_normal = norm_gdn(
                    argument_view_10, w2_linear_attn_norm_weight, argument_view_11, gdn_normal
                )
                argument_view_12 = pl.reshape(gdn_normal, [24, 1536])
                partial = project_attention_out(argument_view_12, bank2_out, partial, group_index)
                argument_view_13 = group_index * 8 + 5
                residual, mlp_norm, data, signal, mlp_quantized, mlp_activation_scale = norm_quant_reduction(
                    partial,
                    hidden,
                    residual,
                    w2_post_attention_layernorm_weight,
                    mlp_norm,
                    data,
                    signal,
                    metadata,
                    argument_view_13,
                    mlp_quantized,
                    mlp_activation_scale,
                )
                mlp_act = mlp_from_quant(
                    mlp_quantized,
                    bank2_gateup,
                    mlp_act,
                    group_index,
                    bank2_gateup_scale,
                    mlp_activation_scale,
                )
                partial = down_projection(mlp_act, bank2_down, partial, group_index, bank2_down_scale)
                next_input_norm2 = pl.reshape(pl.slice(group_weights, [5120], [430464]), [1, 5120])
                argument_view_14 = group_index * 8 + 6
                hidden, norm, data, signal = norm_reduction(
                    partial,
                    residual,
                    hidden,
                    next_input_norm2,
                    norm,
                    data,
                    signal,
                    metadata,
                    argument_view_14,
                )
                w3_input_layernorm_weight = pl.reshape(pl.slice(group_weights, [5120], [430464]), [1, 5120])
                w3_post_attention_layernorm_weight = pl.reshape(
                    pl.slice(group_weights, [5120], [435584]), [1, 5120]
                )
                w3_self_attn_q_norm_weight = pl.reshape(pl.slice(group_weights, [256], [440704]), [1, 256])
                w3_self_attn_k_norm_weight = pl.reshape(pl.slice(group_weights, [256], [440960]), [1, 256])
                indices = pl.reshape(pl.slice(meta3, [24], [0]), [24])
                initial = pl.reshape(pl.slice(meta3, [24], [24]), [24])
                lengths = pl.reshape(pl.slice(meta3, [24], [48]), [24])
                table = pl.reshape(pl.slice(meta3, [8064], [72]), [24, 336])
                packed_q, full_k, full_v = project_qkv(
                    norm, bank3_qkv, packed_q, full_k, full_v, group_index, bank3_qkv_scale
                )
                full_qr, full_gate = norm_rope_q(
                    packed_q, w3_self_attn_q_norm_weight, cos, sin, positions, full_qr, full_gate
                )
                key_flat = pl.reshape(keys3, [294912, 256])
                value_flat = pl.reshape(values3, [294912, 256])
                argument_view_15 = pl.reshape(full_k, [24, 1, 256])
                key_flat, value_flat = norm_rope_k_append(
                    argument_view_15,
                    w3_self_attn_k_norm_weight,
                    cos,
                    sin,
                    positions,
                    full_v,
                    key_flat,
                    value_flat,
                    kv_slots,
                )
                argument_view_16 = pl.reshape(key_flat, [294912, 1, 256])
                attention_scores = compute_scores(full_qr, argument_view_16, table, lengths, attention_scores)
                attention_probability = softmax_scores(
                    attention_scores, lengths, softmax_columns, attention_probability
                )
                argument_view_17 = pl.reshape(value_flat, [294912, 1, 256])
                attention_partials = compute_partial_values(
                    attention_probability, argument_view_17, table, lengths, attention_partials
                )
                gated = reduce_gate(attention_partials, full_gate, gated)
                partial = project_attention_out_w8a8(gated, bank3_out, partial, group_index, bank3_out_scale)
                argument_view_18 = group_index * 8 + 7
                residual, mlp_norm, data, signal, mlp_quantized, mlp_activation_scale = norm_quant_reduction(
                    partial,
                    hidden,
                    residual,
                    w3_post_attention_layernorm_weight,
                    mlp_norm,
                    data,
                    signal,
                    metadata,
                    argument_view_18,
                    mlp_quantized,
                    mlp_activation_scale,
                )
                mlp_act = mlp_from_quant(
                    mlp_quantized,
                    bank3_gateup,
                    mlp_act,
                    group_index,
                    bank3_gateup_scale,
                    mlp_activation_scale,
                )
                partial = down_projection(mlp_act, bank3_down, partial, group_index, bank3_down_scale)
                argument_view_19 = group_index * 8 + 8
                hidden, data, signal = add_reduction(
                    partial, residual, hidden, data, signal, metadata, argument_view_19
                )
            head_norm = norm_hidden(hidden, final_norm, hidden, head_norm)
            hidden_result = publish_hidden(head_norm, hidden_result)
            out = project_head(head_norm, lm_head, out)
            global_logits, logit_data, logit_signal = gather_logits(
                out, global_logits, logit_data, logit_signal, metadata
            )
        else:
            stop_view = pl.reshape(control_i32, [1, 8165])
            stop_view = graph_stop_fence(stop_view)
            control_i32 = pl.reshape(stop_view, [8165])
        return global_logits

    entry = ir.Function(
        f"rank{rank}",
        list(zip(target.params, target.param_directions)),
        target.return_types,
        target.body,
        target.span,
        type=target.func_type,
        level=target.level,
        role=target.role,
        attrs=target.attrs,
        ir_stage=target.ir_stage,
    )
    return ir.Program([*operators, entry], f"MixedW8A8Rank{rank}", target.span)
