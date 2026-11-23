/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Input = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 128>;
using Weight = pto::Tile<pto::TileType::Vec, bfloat16_t, 8, 128>;
using InputFp32 = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Rows = pto::Tile<pto::TileType::Vec, float, 8, 128>;
using Lanes = pto::Tile<pto::TileType::Vec, float, 1, 8>;
using Output = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 16>;
using InputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 2048, 1>>;
using WeightGlobal =
    pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1, 1, 1, 2048, 1>>;
using OutputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned partition = get_block_idx();
  if (partition >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* logits = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + partition * 16;
  if (x->ndims != 2 || x->shapes[0] != 1 || x->shapes[1] != 2048 || weight->ndims != 2 ||
      weight->shapes[0] != 128 || weight->shapes[1] != 2048 || logits->ndims != 2 || logits->shapes[0] != 1 ||
      logits->shapes[1] != 128) {
    PublishStatus(status, 1);
    return;
  }
  Input xb;
  Weight wb;
  InputFp32 xf;
  Rows wf, expanded, products;
  Lanes term, high, low, next, virtualTerm, virtualHigh, errorHigh, errorTerm, error, accumulator;
  Output output;
  pto::TASSIGN<0>(xb);
  pto::TASSIGN<1024>(wb);
  pto::TASSIGN<3072>(xf);
  pto::TASSIGN<4096>(wf);
  pto::TASSIGN<8192>(expanded);
  pto::TASSIGN<12288>(products);
  pto::TASSIGN<16384>(term);
  pto::TASSIGN<16416>(high);
  pto::TASSIGN<16448>(low);
  pto::TASSIGN<16480>(next);
  pto::TASSIGN<16512>(virtualTerm);
  pto::TASSIGN<16544>(virtualHigh);
  pto::TASSIGN<16576>(errorHigh);
  pto::TASSIGN<16608>(errorTerm);
  pto::TASSIGN<16640>(error);
  pto::TASSIGN<16672>(accumulator);
  pto::TASSIGN<16704>(output);
  auto* outputBits = reinterpret_cast<__ubuf__ uint16_t*>(output.data());
  for (unsigned halfIndex = 0; halfIndex < 2; ++halfIndex) {
    pto::TEXPANDS(accumulator, 0.0f);
    pipe_barrier(PIPE_V);
    for (unsigned kb = 0; kb < 16; ++kb) {
      unsigned group = kb / 4;
      if (partition % 2 == 1) group = 3 - group;
      group = (group + partition / 2) % 4;
      const unsigned k0 = group * 512 + (kb % 4) * 128;
      const unsigned n0 = partition * 16 + halfIndex * 8;
      InputGlobal input(Data<bfloat16_t>(x) + k0);
      WeightGlobal weights(Data<bfloat16_t>(weight) + n0 * 2048 + k0);
      pto::TLOAD(xb, input);
      pto::TLOAD(wb, weights);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      pto::TCVT(xf, xb, pto::RoundMode::CAST_NONE);
      pto::TCVT(wf, wb, pto::RoundMode::CAST_NONE);
      pto::TCOLEXPAND(expanded, xf);
      pipe_barrier(PIPE_V);
      pto::TMUL(products, wf, expanded);
      pipe_barrier(PIPE_V);
      for (unsigned base = 0; base < 128; base += 16) {
        pto::TMOV(high, accumulator);
        pto::TEXPANDS(low, 0.0f);
        pipe_barrier(PIPE_V);
        for (unsigned lane = 0; lane < 16; ++lane) {
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          for (unsigned head = 0; head < 8; ++head)
            term.data()[head] = products.data()[head * 128 + base + lane];
          set_flag(PIPE_S, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
          pto::TADD(next, high, term);
          pipe_barrier(PIPE_V);
          pto::TSUB(virtualTerm, next, high);
          pipe_barrier(PIPE_V);
          pto::TSUB(virtualHigh, next, virtualTerm);
          pipe_barrier(PIPE_V);
          pto::TSUB(errorHigh, high, virtualHigh);
          pto::TSUB(errorTerm, term, virtualTerm);
          pipe_barrier(PIPE_V);
          pto::TADD(error, errorHigh, errorTerm);
          pipe_barrier(PIPE_V);
          pto::TADD(low, low, error);
          pto::TMOV(high, next);
          pipe_barrier(PIPE_V);
        }
        pto::TADD(accumulator, high, low);
        pipe_barrier(PIPE_V);
      }
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    auto* resultBits = reinterpret_cast<__ubuf__ uint32_t*>(accumulator.data());
    for (unsigned head = 0; head < 8; ++head)
      outputBits[halfIndex * 8 + head] = pto::detail::A1Fp32ToBf16Bits(resultBits[head]);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  OutputGlobal result(Data<bfloat16_t>(logits) + partition * 16);
  pto::TSTORE(result, output);
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
