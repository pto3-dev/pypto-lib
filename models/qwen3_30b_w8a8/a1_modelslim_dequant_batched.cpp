/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#define PYPTO_A1_BATCHED_VECTOR_PADDING 1
#include "a1_modelslim_expert_dequant.h"

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  constexpr unsigned blocks = 8;
  const unsigned block = get_block_idx();
  if (block >= blocks) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* bias = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != blocks || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + block * 16;
  const unsigned width = x->shapes[1];
  if (x->ndims != 2 || x->shapes[0] != 32 || (width != 2048 && width != 5120) ||
      bias->ndims != 2 || bias->shapes[0] != 1 || bias->shapes[1] != width ||
      scale->ndims != 2 || scale->shapes[0] != 1 || scale->shapes[1] != width ||
      output->ndims != 2 || output->shapes[0] != 20 || output->shapes[1] != width) {
    PublishStatus(status, 1);
    return;
  }
  using Integers = pto::Tile<pto::TileType::Vec, int32_t, 1, 128>;
  using Scales = pto::Tile<pto::TileType::Vec, float, 1, 128>;
  using Halves = pto::Tile<pto::TileType::Vec, half, 1, 128>;
  using GlobalInt = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  using GlobalFloat = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  using GlobalHalf = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  Integers integers, biases;
  Scales scales;
  Halves rounded;
  pto::TASSIGN<0>(integers);
  pto::TASSIGN<512>(biases);
  pto::TASSIGN<1024>(scales);
  pto::TASSIGN<1536>(rounded);
  for (unsigned task = block; task < 20 * (width / 128); task += blocks) {
    const unsigned row = task / (width / 128);
    const unsigned col = task % (width / 128) * 128;
    GlobalInt input(Data<int32_t>(x) + row * width + col);
    GlobalInt channelBias(Data<int32_t>(bias) + col);
    GlobalFloat channelScale(Data<float>(scale) + col);
    pto::TLOAD(integers, input);
    pto::TLOAD(biases, channelBias);
    pto::TLOAD(scales, channelScale);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (unsigned lane = 0; lane < 128; ++lane) integers.data()[lane] += biases.data()[lane];
    pipe_barrier(PIPE_ALL);
    ModelSlimChannelDequant(integers, scales, rounded);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GlobalHalf destination(Data<half>(output) + row * width + col);
    pto::TSTORE(destination, rounded);
    pipe_barrier(PIPE_ALL);
  }
  PublishStatus(status, 0);
}
