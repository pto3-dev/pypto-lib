/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Half = pto::Tile<pto::TileType::Vec, half, 1, 128>;
using Float = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Integer = pto::Tile<pto::TileType::Vec, int32_t, 1, 128>;
using Quant = pto::Tile<pto::TileType::Vec, int8_t, 1, 128>;
using HalfGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using QuantGlobal = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[3]);
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  if (x->ndims != 2 || (x->shapes[0] != 1 && x->shapes[0] != 8 && x->shapes[0] != 20 && x->shapes[0] != 160) ||
      (x->shapes[1] != 2048 && x->shapes[1] != 768) || quant->ndims != 2 ||
      quant->shapes[0] != x->shapes[0] || quant->shapes[1] != x->shapes[1] ||
      scale->ndims != 2 || scale->shapes[0] != x->shapes[0] || scale->shapes[1] != 16 ||
      status->ndims != 2 || status->shapes[0] != x->shapes[0] || status->shapes[1] != 16) {
    trap();
    return;
  }
  const unsigned width = x->shapes[1];
  Half input, integral;
  Float values;
  Integer rounded;
  Quant output;
  pto::TASSIGN<0>(input);
  pto::TASSIGN<256>(values);
  pto::TASSIGN<768>(rounded);
  pto::TASSIGN<1280>(integral);
  pto::TASSIGN<1536>(output);
  for (unsigned row = block; row < x->shapes[0]; row += 8) {
  float maximum = 0.0f;
  float inverse = 0.0f;
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (unsigned col = 0; col < width; col += 128) {
      HalfGlobal source(Data<half>(x) + row * width + col);
      pto::TLOAD(input, source);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(values, input, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      if (pass == 0) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (unsigned lane = 0; lane < 128; ++lane) {
          const float value = values.data()[lane];
          const float magnitude = value < 0.0f ? -value : value;
          if (magnitude > maximum) maximum = magnitude;
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      } else {
        pto::TMULS(values, values, inverse);
        pipe_barrier(PIPE_V);
        pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        pto::TCVT(integral, values, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(output, integral, pto::RoundMode::CAST_TRUNC);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        QuantGlobal destination(Data<int8_t>(quant) + row * width + col);
        pto::TSTORE(destination, output);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    if (pass == 0) {
      const float dequant = maximum * (1.0f / 127.0f);
      auto* destination = Data<float>(scale) + row * 16;
      destination[0] = dequant;
      dcci((__gm__ void*)destination, cache_line_t::SINGLE_CACHE_LINE);
      inverse = dequant > 1e-7f ? 1.0f / dequant : 0.0f;
    }
  }
  PublishStatus(Data<int32_t>(status) + row * 16, 0);
  }
}
