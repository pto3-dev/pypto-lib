/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_modelslim_expert_dequant.h"

// Observable expert-GMM materialization, also used to isolate precision failures.
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* acc = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* weightScale = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
  const unsigned width = acc->shapes[1];
  if (acc->ndims != 2 || acc->shapes[0] != 160 || (width != 1536 && width != 2048) ||
      inputScale->ndims != 2 || (inputScale->shapes[0] != 20 && inputScale->shapes[0] != 160) || inputScale->shapes[1] != 16 ||
      ids->ndims != 2 || ids->shapes[0] != 20 || ids->shapes[1] != 8 ||
      weightScale->ndims != 2 || weightScale->shapes[0] != 128 || weightScale->shapes[1] != width ||
      output->ndims != 2 || output->shapes[0] != 160 || output->shapes[1] != width ||
      statusTensor->ndims != 2 || statusTensor->shapes[0] != 160 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  using Integer = pto::Tile<pto::TileType::Vec, int32_t, 1, 128>;
  using Float = pto::Tile<pto::TileType::Vec, float, 1, 128>;
  using Half = pto::Tile<pto::TileType::Vec, half, 1, 128>;
  using IntegerGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  using FloatGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  using HalfGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  Integer integer;
  Float scale, value;
  Half rounded;
  pto::TASSIGN<0>(integer);
  pto::TASSIGN<512>(scale);
  pto::TASSIGN<1024>(value);
  pto::TASSIGN<1536>(rounded);
  for (unsigned route = block; route < 160; route += 8) {
    const int32_t expert = Data<int32_t>(ids)[route];
    auto* status = Data<int32_t>(statusTensor) + route * 16;
    if (expert < 0 || expert >= 128) {
      PublishStatus(status, 2);
      continue;
    }
    const float activationScale = Data<float>(inputScale)[(inputScale->shapes[0] == 20 ? route / 8 : route) * 16];
    for (unsigned col = 0; col < width; col += 128) {
      IntegerGlobal source(Data<int32_t>(acc) + route * width + col);
      FloatGlobal channel(Data<float>(weightScale) + expert * width + col);
      pto::TLOAD(integer, source);
      pto::TLOAD(scale, channel);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      ModelSlimChannelDequant(integer, scale, rounded);
      pto::TCVT(value, rounded, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TMULS(value, value, activationScale);
      pipe_barrier(PIPE_V);
      pto::TCVT(rounded, value, pto::RoundMode::CAST_RINT);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      HalfGlobal destination(Data<half>(output) + route * width + col);
      pto::TSTORE(destination, rounded);
      pipe_barrier(PIPE_ALL);
    }
    PublishStatus(status, 0);
  }
}
