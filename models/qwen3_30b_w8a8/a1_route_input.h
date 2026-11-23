/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

template <unsigned Width, unsigned SourceRows>
__aicore__ inline void RouteInput(__gm__ int64_t* args) {
  const unsigned route = get_block_idx();
  if (route >= 8) return;
  auto* input = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* outputScale = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + route * 16;
  if (input->ndims != 2 || input->shapes[0] != SourceRows || input->shapes[1] != Width ||
      inputScale->ndims != 2 || inputScale->shapes[0] != SourceRows || inputScale->shapes[1] != 16 ||
      output->ndims != 2 || output->shapes[0] != 128 || output->shapes[1] != Width ||
      outputScale->ndims != 2 || outputScale->shapes[0] != 8 || outputScale->shapes[1] != 16) {
    PublishStatus(status, 1);
    return;
  }
  using Row = pto::Tile<pto::TileType::Vec, int8_t, 1, 128>;
  using Global = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, Width, 1>>;
  Row values, zeros;
  pto::TASSIGN<0>(values);
  pto::TASSIGN<128>(zeros);
  for (unsigned lane = 0; lane < 128; ++lane) zeros.data()[lane] = 0;
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  const unsigned sourceRow = SourceRows == 1 ? 0 : route;
  for (unsigned col = 0; col < Width; col += 128) {
    Global source(Data<int8_t>(input) + sourceRow * Width + col);
    pto::TLOAD(values, source);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    Global active(Data<int8_t>(output) + route * 16 * Width + col);
    pto::TSTORE(active, values);
    for (unsigned row = 1; row < 16; ++row) {
      Global padding(Data<int8_t>(output) + (route * 16 + row) * Width + col);
      pto::TSTORE(padding, zeros);
    }
    pipe_barrier(PIPE_ALL);
  }
  auto* scaleOutput = Data<float>(outputScale) + route * 16;
  scaleOutput[0] = Data<float>(inputScale)[sourceRow * 16];
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)scaleOutput, cache_line_t::SINGLE_CACHE_LINE);
  PublishStatus(status, 0);
}
