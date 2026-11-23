/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Input = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 2048>;
using ExpandedInput = pto::Tile<pto::TileType::Vec, float, 1, 2048>;
using Row = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Rounded = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 128>;
using InputGlobal =
    pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<1, 1, 1, 2048, 1>>;
using RowGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned partition = get_block_idx();
  if (partition >= 8) return;
  auto* input = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + partition * 16;
  const int64_t width = weight->shapes[1];
  if (input->ndims != 2 || input->shapes[0] != 1 || input->shapes[1] != 2048 || weight->ndims != 2 ||
      weight->shapes[0] != 2048 || width <= 0 || width % 128 != 0 || output->ndims != 2 ||
      output->shapes[0] != 1 || output->shapes[1] != width) {
    PublishStatus(status, 1);
    return;
  }
  Input xb;
  ExpandedInput xf;
  Row weights, term, high, low, next, virtualTerm, virtualHigh, errorHigh, errorTerm, error, accumulator,
      result;
  Rounded rounded;
  pto::TASSIGN<0>(xb);
  pto::TASSIGN<4096>(xf);
  pto::TASSIGN<12288>(weights);
  pto::TASSIGN<12800>(term);
  pto::TASSIGN<13312>(high);
  pto::TASSIGN<13824>(low);
  pto::TASSIGN<14336>(next);
  pto::TASSIGN<14848>(virtualTerm);
  pto::TASSIGN<15360>(virtualHigh);
  pto::TASSIGN<15872>(errorHigh);
  pto::TASSIGN<16384>(errorTerm);
  pto::TASSIGN<16896>(error);
  pto::TASSIGN<17408>(accumulator);
  pto::TASSIGN<17920>(rounded);
  pto::TASSIGN<18176>(result);
  InputGlobal source(Data<bfloat16_t>(input));
  pto::TLOAD(xb, source);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(xf, xb, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int64_t column = partition * 128; column < width; column += 8 * 128) {
    pto::TEXPANDS(accumulator, 0.0f);
    pipe_barrier(PIPE_V);
    for (unsigned base = 0; base < 2048; base += 16) {
      pto::TMOV(high, accumulator);
      pto::TEXPANDS(low, 0.0f);
      pipe_barrier(PIPE_V);
      for (unsigned lane = 0; lane < 16; ++lane) {
        const unsigned k = base + lane;
        RowGlobal row(Data<float>(weight) + k * width + column);
        pto::TLOAD(weights, row);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        const float scalar = xf.data()[k];
        pto::TMULS(term, weights, scalar);
        pipe_barrier(PIPE_V);
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
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      }
      pto::TADD(accumulator, high, low);
      pipe_barrier(PIPE_V);
    }
    pto::TCVT(rounded, accumulator, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(result, rounded, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    RowGlobal target(Data<float>(output) + column);
    pto::TSTORE(target, result);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
