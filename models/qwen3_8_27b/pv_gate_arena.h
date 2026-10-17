/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once
#include <pto/pto-inst.hpp>
namespace pv_gate {
using F=pto::Tile<pto::TileType::Vec,float,1,1536,pto::BLayout::RowMajor,1,1536>;
using B=pto::Tile<pto::TileType::Vec,bfloat16_t,1,1536,pto::BLayout::RowMajor,1,1536>;
template<class T> using G=pto::GlobalTensor<T,pto::Shape<1,1,1,1,1536>,pto::Stride<1536,1536,1536,1536,1>>;
// TPipe owns UB [0,32768); the generated BF16 PV output owns [32768,40960).
struct Arena {
 B gate_b,sigmoid_b,core_b,result_b;
 F gate_f,sigmoid_f,core_f,result_f;
 __aicore__ Arena() {
  pto::TASSIGN(gate_b,49152);pto::TASSIGN(gate_f,53248);
  pto::TASSIGN(sigmoid_b,61440);pto::TASSIGN(sigmoid_f,65536);
  pto::TASSIGN(core_b,32768);pto::TASSIGN(core_f,73728);
  pto::TASSIGN(result_f,81920);pto::TASSIGN(result_b,90112);
 }
};
static __aicore__ __attribute__((noinline)) void prefetch(Arena& a,__gm__ bfloat16_t* gate) {
 G<bfloat16_t> input(gate);pto::TLOAD(a.gate_b,input);
 set_flag(PIPE_MTE2,PIPE_V,EVENT_ID4);wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID4);
 pto::TCVT(a.gate_f,a.gate_b,pto::RoundMode::CAST_ROUND);
 pipe_barrier(PIPE_V);pto::TNEG(a.gate_f,a.gate_f);
 pipe_barrier(PIPE_V);pto::TEXP(a.gate_f,a.gate_f);
 pipe_barrier(PIPE_V);pto::TADDS(a.gate_f,a.gate_f,1.0f);
 pipe_barrier(PIPE_V);pto::TRECIP(a.sigmoid_f,a.gate_f);
 pipe_barrier(PIPE_V);pto::TCVT(a.sigmoid_b,a.sigmoid_f,pto::RoundMode::CAST_RINT);
 pipe_barrier(PIPE_V);pto::TCVT(a.sigmoid_f,a.sigmoid_b,pto::RoundMode::CAST_ROUND);
}
static __aicore__ __attribute__((noinline)) void finish(Arena& a,__gm__ bfloat16_t* gated) {
 pipe_barrier(PIPE_V);pto::TCVT(a.core_f,a.core_b,pto::RoundMode::CAST_ROUND);
 pipe_barrier(PIPE_V);pto::TMUL(a.result_f,a.sigmoid_f,a.core_f);
 pipe_barrier(PIPE_V);pto::TCVT(a.result_b,a.result_f,pto::RoundMode::CAST_RINT);
 set_flag(PIPE_V,PIPE_MTE3,EVENT_ID4);wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID4);
 G<bfloat16_t> output(gated);pto::TSTORE(output,a.result_b);
 set_flag(PIPE_MTE3,PIPE_V,EVENT_ID4);wait_flag(PIPE_MTE3,PIPE_V,EVENT_ID4);
}
}
