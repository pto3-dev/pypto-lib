// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#include "a1_tensor.h"

namespace {
constexpr unsigned Width = 2048;
constexpr unsigned HalfBytes = Width * sizeof(half);
constexpr unsigned FloatBytes = Width * sizeof(float);
constexpr unsigned InputBase = 0;
constexpr unsigned SkipBase = InputBase + HalfBytes;
constexpr unsigned GammaHalfBase = SkipBase + HalfBytes;
constexpr unsigned BetaHalfBase = GammaHalfBase + HalfBytes;
constexpr unsigned ValuesBase = BetaHalfBase + HalfBytes;
constexpr unsigned AuxiliaryBase = ValuesBase + FloatBytes;
constexpr unsigned SquareBase = AuxiliaryBase + FloatBytes;
constexpr unsigned GammaBase = SquareBase + FloatBytes;
constexpr unsigned BetaBase = GammaBase + FloatBytes;
constexpr unsigned PartialBase = BetaBase + FloatBytes;
constexpr unsigned VarianceBase = PartialBase + 64 * sizeof(float);
constexpr unsigned RootBase = VarianceBase + 8 * sizeof(float);
constexpr unsigned OnesBase = RootBase + 8 * sizeof(float);
constexpr unsigned InverseBase = OnesBase + 8 * sizeof(float);
constexpr unsigned RoundedBase = InverseBase + 8 * sizeof(float);
static_assert(RoundedBase % 32 == 0 && RoundedBase + HalfBytes <= 64 * 1024);
using Half = pto::Tile<pto::TileType::Vec, half, 1, Width>;
using Float = pto::Tile<pto::TileType::Vec, float, 1, Width>;
using Partial = pto::Tile<pto::TileType::Vec, float, 1, 64>;
using Scalar = pto::Tile<pto::TileType::Vec, float, 1, 8>;
using Global = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, Width>,
                               pto::Stride<1, 1, 1, Width, 1>>;
}

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* skip = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* gamma = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* beta = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* y = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* skipOut = reinterpret_cast<__gm__ Tensor*>(args[5]);
  for (unsigned index = 0; index < 6; ++index) {
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    const unsigned rows = index == 2 || index == 3 ? 1 : 20;
    if (tensor->ndims != 2 || tensor->shapes[0] != rows || tensor->shapes[1] != Width) {
      trap();
      return;
    }
  }
  Half input, skipHalf, gammaHalf, betaHalf, rounded;
  Float values, auxiliary, square, gammaFloat, betaFloat;
  Partial partial;
  Scalar variance, root, ones, inverse;
  pto::TASSIGN<InputBase>(input);
  pto::TASSIGN<SkipBase>(skipHalf);
  pto::TASSIGN<GammaHalfBase>(gammaHalf);
  pto::TASSIGN<BetaHalfBase>(betaHalf);
  pto::TASSIGN<RoundedBase>(rounded);
  pto::TASSIGN<ValuesBase>(values);
  pto::TASSIGN<AuxiliaryBase>(auxiliary);
  pto::TASSIGN<SquareBase>(square);
  pto::TASSIGN<GammaBase>(gammaFloat);
  pto::TASSIGN<BetaBase>(betaFloat);
  pto::TASSIGN<PartialBase>(partial);
  pto::TASSIGN<VarianceBase>(variance);
  pto::TASSIGN<RootBase>(root);
  pto::TASSIGN<OnesBase>(ones);
  pto::TASSIGN<InverseBase>(inverse);
  Global gammaSource(Data<half>(gamma)), betaSource(Data<half>(beta));
  pto::TLOAD(gammaHalf, gammaSource);
  pto::TLOAD(betaHalf, betaSource);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(gammaFloat, gammaHalf, pto::RoundMode::CAST_NONE);
  pto::TCVT(betaFloat, betaHalf, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  for (unsigned row = block; row < 20; row += 8) {
    Global source(Data<half>(x) + row * Width);
    Global residual(Data<half>(skip) + row * Width);
    pto::TLOAD(input, source);
    pto::TLOAD(skipHalf, residual);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(values, input, pto::RoundMode::CAST_NONE);
    pto::TCVT(auxiliary, skipHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TADD(values, values, auxiliary);
    pipe_barrier(PIPE_V);
    pto::TCVT(input, values, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Global residualDestination(Data<half>(skipOut) + row * Width);
    pto::TSTORE(residualDestination, input);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::TCVT(values, input, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(square, values, values);
    pipe_barrier(PIPE_V);
    pto::TMULS(square, square, 1.0f / Width);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vadds(partial.data(), square.data(), 0.0f, 1, 1, 1, 8, 8);
    // Preserve the DSL's ascending 32-chunk fold before its 64-lane reduction.
    for (unsigned column = 64; column < Width; column += 64) {
      pipe_barrier(PIPE_V);
      vadd(partial.data(), partial.data(), square.data() + column, 1, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    pto::TEXPANDS(variance, 0.0f);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, UINT64_MAX);
    vcadd(variance.data(), partial.data(), 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    pto::TADDS(variance, variance, 1e-6f);
    pipe_barrier(PIPE_V);
    pto::TSQRT(root, variance);
    pipe_barrier(PIPE_V);
    pto::TMULS(ones, root, 0.0f);
    pipe_barrier(PIPE_V);
    pto::TADDS(ones, ones, 1.0f);
    pipe_barrier(PIPE_V);
    pto::TDIV(inverse, ones, root);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float factor = inverse.data()[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    pto::TMULS(values, values, factor);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(values, values, gammaFloat);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TADD(values, values, betaFloat);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Global destination(Data<half>(y) + row * Width);
    pto::TSTORE(destination, rounded);
    pipe_barrier(PIPE_ALL);
  }
}
