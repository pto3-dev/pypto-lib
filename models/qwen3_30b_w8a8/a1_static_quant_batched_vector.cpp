/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"
#include "a1_bounded_int8_pack.h"

template <unsigned Cols>
__aicore__ inline void StaticQuantColumns(__gm__ Tensor* x, __gm__ Tensor* reciprocal,
                                        __gm__ Tensor* offset, __gm__ Tensor* quant,
                                        unsigned block) {
  static_assert(Cols == 256 || Cols == 512);
  constexpr unsigned HalfBytes = Cols * sizeof(half);
  constexpr unsigned FloatBytes = Cols * sizeof(float);
  constexpr unsigned InverseHalfBase = HalfBytes;
  constexpr unsigned ZeroHalfBase = InverseHalfBase + HalfBytes;
  constexpr unsigned ValuesBase = ZeroHalfBase + HalfBytes;
  constexpr unsigned InverseBase = ValuesBase + FloatBytes;
  constexpr unsigned ZeroBase = InverseBase + FloatBytes;
  constexpr unsigned RoundedBase = ZeroBase + FloatBytes;
  constexpr unsigned IntegralBase = RoundedBase + FloatBytes;
  constexpr unsigned QuantBase = IntegralBase + HalfBytes;
  constexpr unsigned CeilingBase = QuantBase + Cols;
  constexpr unsigned FloorBase = CeilingBase + FloatBytes;
  static_assert(FloorBase + FloatBytes <= 32 * 1024);
  using Half = pto::Tile<pto::TileType::Vec, half, 1, Cols>;
  using Float = pto::Tile<pto::TileType::Vec, float, 1, Cols>;
  using Integer = pto::Tile<pto::TileType::Vec, int32_t, 1, Cols>;
  using Quant = pto::Tile<pto::TileType::Vec, int8_t, 1, Cols>;
  using HalfGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
  using QuantGlobal = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
  Half input, inverseHalf, zeroHalf, integral;
  Float values, inverse, zero, ceiling, floor;
  Integer rounded;
  Quant result;
  pto::TASSIGN<0>(input);
  pto::TASSIGN<InverseHalfBase>(inverseHalf);
  pto::TASSIGN<ZeroHalfBase>(zeroHalf);
  pto::TASSIGN<ValuesBase>(values);
  pto::TASSIGN<InverseBase>(inverse);
  pto::TASSIGN<ZeroBase>(zero);
  pto::TASSIGN<RoundedBase>(rounded);
  pto::TASSIGN<IntegralBase>(integral);
  pto::TASSIGN<QuantBase>(result);
  pto::TASSIGN<CeilingBase>(ceiling);
  pto::TASSIGN<FloorBase>(floor);
  pto::TEXPANDS(ceiling, 127.0f);
  pto::TEXPANDS(floor, -128.0f);
  pipe_barrier(PIPE_V);
  constexpr unsigned width = 8 * Cols;
  const unsigned column = block * Cols;
  HalfGlobal inverseSource(Data<half>(reciprocal) + column);
  HalfGlobal zeroSource(Data<half>(offset) + column);
  pto::TLOAD(inverseHalf, inverseSource);
  pto::TLOAD(zeroHalf, zeroSource);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(inverse, inverseHalf, pto::RoundMode::CAST_NONE);
  pto::TCVT(zero, zeroHalf, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  for (unsigned row = 0; row < 20; ++row) {
    HalfGlobal source(Data<half>(x) + row * width + column);
    pto::TLOAD(input, source);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(values, input, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(values, values, inverse);
    pipe_barrier(PIPE_V);
    pto::TADD(values, values, zero);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vmin(values.data(), values.data(), ceiling.data(), Cols / 64, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmax(values.data(), values.data(), floor.data(), Cols / 64, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(integral, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    PackBoundedIntegralInt8(result, integral);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    QuantGlobal destination(Data<int8_t>(quant) + row * width + column);
    pto::TSTORE(destination, result);
    pipe_barrier(PIPE_ALL);
  }
}

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* reciprocal = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* offset = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[3]);
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  if (x->ndims != 2 || x->shapes[0] != 20 || (x->shapes[1] != 2048 && x->shapes[1] != 4096) ||
      reciprocal->ndims != 2 || reciprocal->shapes[0] != 1 || reciprocal->shapes[1] != x->shapes[1] ||
      offset->ndims != 2 || offset->shapes[0] != 1 || offset->shapes[1] != x->shapes[1] ||
      quant->ndims != 2 || quant->shapes[0] != 20 || quant->shapes[1] != x->shapes[1]) {
    trap();
    return;
  }
  if (x->shapes[1] == 2048) StaticQuantColumns<256>(x, reciprocal, offset, quant, block);
  else StaticQuantColumns<512>(x, reciprocal, offset, quant, block);
}
