# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .down_i32 import project_down_i32_kernel,packed_mlp_i32_from_quant
from .collective_dequant_row import OPERATORS as DEQUANT_OPERATORS
from .packed_act_quant import packed_gateup_kernel, swiglu_quant_kernel, packed_mlp_from_quant
"""Explicit IR link order; keeps legacy function order for reproducible lowering."""
from .attention import (split_query_kernel, split_query, norm_rope_q_kernel, norm_rope_q, norm_rope_k_append_kernel, norm_rope_k_append, compute_scores)
from .collectives import (reduce_embedding, reduce_partial, gather_logits_kernel, gather_logits, reduce_add_kernel, reduce_add, reduce_add_norm_kernel, reduce_add_norm, publish_hidden, graph_stop_fence, reduce_add_norm_quant_kernel, reduce_add_norm_quant)
from .embedding import (embedding0_kernel, embedding0, embedding1_kernel, embedding1, embedding2_kernel, embedding2, embedding3_kernel, embedding3)
from .gdn import (convolve, recurrent_kernel, recurrent, gates_kernel, gates)
from .norm_pointwise import (norm_plain_kernel, norm_plain, norm_hidden_kernel, norm_hidden, norm_gdn_kernel, norm_gdn, add_kernel, add, activate_kernel, activate, gate_attention_kernel, gate_attention)
from .projection_bf16 import (greedy_projection_spmd_kernel, project_attention_out, project_gdn_kernel, project_gdn, project_z_kernel, project_z, project_up_kernel, project_up, project_head_kernel, project_head)
from .quantized import (mlp_from_quant, quant_5120, quant_4352, quant_1536, mlp_fused_kernel, mlp_fused, project_down_kernel, project_down, project_qkv_kernel, project_qkv, project_attention_out_w8a8_kernel, project_attention_out_w8a8)

OPERATORS = (*DEQUANT_OPERATORS,project_down_i32_kernel,packed_mlp_i32_from_quant,
    packed_gateup_kernel,
    swiglu_quant_kernel,
    packed_mlp_from_quant,
    reduce_embedding,
    reduce_partial,
    gather_logits_kernel,
    gather_logits,
    reduce_add_kernel,
    reduce_add,
    reduce_add_norm_kernel,
    reduce_add_norm,
    greedy_projection_spmd_kernel,
    project_attention_out,
    norm_plain_kernel,
    norm_plain,
    norm_hidden_kernel,
    norm_hidden,
    norm_gdn_kernel,
    norm_gdn,
    add_kernel,
    add,
    activate_kernel,
    activate,
    gate_attention_kernel,
    gate_attention,
    project_gdn_kernel,
    project_gdn,
    project_z_kernel,
    project_z,
    project_up_kernel,
    project_up,
    project_head_kernel,
    project_head,
    convolve,
    recurrent_kernel,
    recurrent,
    gates_kernel,
    gates,
    split_query_kernel,
    split_query,
    norm_rope_q_kernel,
    norm_rope_q,
    norm_rope_k_append_kernel,
    norm_rope_k_append,
    compute_scores,
    embedding0_kernel,
    embedding0,
    embedding1_kernel,
    embedding1,
    embedding2_kernel,
    embedding2,
    embedding3_kernel,
    embedding3,
    publish_hidden,
    graph_stop_fence,
    reduce_add_norm_quant_kernel,
    reduce_add_norm_quant,
    mlp_from_quant,
    quant_5120,
    quant_4352,
    quant_1536,
    mlp_fused_kernel,
    mlp_fused,
    project_down_kernel,
    project_down,
    project_qkv_kernel,
    project_qkv,
    project_attention_out_w8a8_kernel,
    project_attention_out_w8a8,
)

# Source-level functions supplied to ir.Program. Normal compilation owns all
# physical kernel registration and metadata; there is no binary relinking.
from .attention_fd_plan import fd_plan_kernel,fd_plan
from .attention_fd import fd_softmax_kernel,fd_softmax,fd_partial_values,fd_combine_kernel,fd_combine
OPERATORS = (*OPERATORS,fd_plan_kernel,fd_plan,fd_softmax_kernel,fd_softmax,
             fd_partial_values,fd_combine_kernel,fd_combine)
