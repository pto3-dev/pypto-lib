/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"
#include "a1_bounded_int8_pack.h"

template <unsigned Width>
__aicore__ inline void RetainedRowQuant(__gm__ int64_t* args) {
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[3]);
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  using Half = pto::Tile<pto::TileType::Vec, half, 1, Width>;
  using Float = pto::Tile<pto::TileType::Vec, float, 1, Width>;
  using Integer = pto::Tile<pto::TileType::Vec, int32_t, 1, Width>;
  using Quant = pto::Tile<pto::TileType::Vec, int8_t, 1, Width>;
  using Small = pto::Tile<pto::TileType::Vec, float, 1, 128>;
  using Input = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, Width>, pto::Stride<1, 1, 1, Width, 1>>;
  using Output = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, Width>, pto::Stride<1, 1, 1, Width, 1>>;
  Half input, integral;
  Float values;
  Integer rounded;
  Quant output;
  Small maximumLanes, magnitude, halves, reduced;
  pto::TASSIGN<0>(input);
  pto::TASSIGN<4096>(values);
  pto::TASSIGN<12288>(rounded);
  pto::TASSIGN<20480>(integral);
  pto::TASSIGN<24576>(output);
  pto::TASSIGN<32768>(maximumLanes);
  pto::TASSIGN<33280>(magnitude);
  pto::TASSIGN<33792>(halves);
  pto::TASSIGN<34304>(reduced);
  for (unsigned row = block; row < x->shapes[0]; row += 8) {
    Input source(Data<half>(x) + row * Width);
    pto::TLOAD(input, source);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(values, input, pto::RoundMode::CAST_NONE);
    pto::TEXPANDS(maximumLanes, 0.0f);
    pipe_barrier(PIPE_V);
    for (unsigned column = 0; column < Width; column += 128) {
      Small chunk;
      pto::TASSIGN(chunk, reinterpret_cast<uint64_t>(values.data() + column));
      pto::TABS(magnitude, chunk);
      pipe_barrier(PIPE_V);
      set_mask_norm();
      set_vector_mask(0, UINT64_MAX);
      vmax(maximumLanes.data(), maximumLanes.data(), magnitude.data(), 2, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vmax(halves.data(), maximumLanes.data(), maximumLanes.data() + 64, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcmax(reduced.data(), halves.data(), 1, 1, 1, 8, false, false, false);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // Keep the original input-quant multiplication by the rounded reciprocal.
    const float dequant = reduced.data()[0] * (1.0f / 127.0f);
    auto* destinationScale = Data<float>(scale) + row * 16;
    destinationScale[0] = dequant;
    dcci((__gm__ void*)destinationScale, cache_line_t::SINGLE_CACHE_LINE);
    const float inverse = dequant > 1e-7f ? 1.0f / dequant : 0.0f;
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    pto::TMULS(values, values, inverse);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(integral, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    PackBoundedIntegralInt8(output, integral);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Output target(Data<int8_t>(quant) + row * Width);
    pto::TSTORE(target, output);
    pipe_barrier(PIPE_ALL);
    PublishStatus(Data<int32_t>(status) + row * 16, 0);
  }
}

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (x->ndims != 2 || (x->shapes[0] != 1 && x->shapes[0] != 8 && x->shapes[0] != 20 && x->shapes[0] != 160) ||
      (x->shapes[1] != 2048 && x->shapes[1] != 768) || quant->ndims != 2 ||
      quant->shapes[0] != x->shapes[0] || quant->shapes[1] != x->shapes[1] ||
      scale->ndims != 2 || scale->shapes[0] != x->shapes[0] || scale->shapes[1] != 16 ||
      status->ndims != 2 || status->shapes[0] != x->shapes[0] || status->shapes[1] != 16) {
    trap();
    return;
  }
  if (x->shapes[1] == 2048) RetainedRowQuant<2048>(args);
  else RetainedRowQuant<768>(args);
}
