/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using IntegerRow = pto::Tile<pto::TileType::Vec, int32_t, 1, 128>;
using FloatRow = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using HalfRow = pto::Tile<pto::TileType::Vec, half, 1, 128>;
using IntegerMatrix = pto::Tile<pto::TileType::Vec, int32_t, 16, 16>;
using HalfMatrix = pto::Tile<pto::TileType::Vec, half, 16, 16>;
using HalfPanel = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using ScaleWords = pto::Tile<pto::TileType::Vec, int32_t, 1, 32>;
using Acc = pto::Tile<pto::TileType::Acc, int32_t, 16, 16, pto::BLayout::ColMajor,
                      16, 16, pto::SLayout::RowMajor, 1024>;
using IntegerGlobal =
    pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using FloatGlobal =
    pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using HalfGlobal =
    pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned partition = get_block_idx();
  if (partition >= 8) return;
  auto* accumulator = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* bias = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + partition * 16;
  const int64_t width = accumulator->shapes[1];
  if (accumulator->ndims != 2 || accumulator->shapes[0] != 1 ||
      (width != 2048 && width != 5120) || bias->ndims != 2 || bias->shapes[0] != 1 ||
      bias->shapes[1] != width || scale->ndims != 2 || scale->shapes[0] != 1 ||
      scale->shapes[1] != width || output->ndims != 2 || output->shapes[0] != 1 ||
      output->shapes[1] != width) {
    PublishStatus(status, 1);
    return;
  }
  IntegerRow integers, biases;
  FloatRow scales;
  IntegerMatrix matrix;
  HalfMatrix rounded;
  HalfPanel active;
  ScaleWords scaleWords;
  Acc l0c;
  pto::TASSIGN<0>(integers);
  pto::TASSIGN<512>(biases);
  pto::TASSIGN<1536>(scales);
  pto::TASSIGN<2048>(matrix);
  pto::TASSIGN<3072>(rounded);
  pto::TASSIGN<3072>(active);
  pto::TASSIGN<3584>(scaleWords);
  pto::TASSIGN<0>(l0c);
  for (int64_t column = partition * 128; column < width; column += 8 * 128) {
    IntegerGlobal source(Data<int32_t>(accumulator) + column);
    IntegerGlobal biasSource(Data<int32_t>(bias) + column);
    FloatGlobal scaleSource(Data<float>(scale) + column);
    pto::TLOAD(integers, source);
    pto::TLOAD(biases, biasSource);
    pto::TLOAD(scales, scaleSource);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (unsigned panel = 0; panel < 128; panel += 16) {
      for (unsigned element = 0; element < 256; ++element) matrix.data()[element] = 0;
      auto* words = reinterpret_cast<__ubuf__ uint32_t*>(scaleWords.data());
      auto* rawScale = reinterpret_cast<__ubuf__ uint32_t*>(scales.data());
      for (unsigned lane = 0; lane < 16; ++lane) {
        matrix.data()[lane] = integers.data()[panel + lane] + biases.data()[panel + lane];
        words[2 * lane] = rawScale[panel + lane];
        words[2 * lane + 1] = 0;
      }
      pipe_barrier(PIPE_ALL);
      copy_matrix_ubuf_to_cc(l0c.data(), matrix.data(), 0, 1, 1, 0, 0, CRMODE_NONE);
      pipe_barrier(PIPE_ALL);
      set_deqscale(reinterpret_cast<uint64_t>(scaleWords.data()) >> 5);
      copy_matrix_cc_to_ubuf(rounded.data(), l0c.data(), 0, 1, 1, 0, 0, CRMODE_DEQSCALE_VDEQ16);
      pipe_barrier(PIPE_ALL);
      HalfGlobal destination(Data<half>(output) + column + panel);
      pto::TSTORE(destination, active);
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
  }
  PublishStatus(status, 0);
}
