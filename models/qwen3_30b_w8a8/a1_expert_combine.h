/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"
#ifdef PYPTO_A1_MODEL_FP16
#include "a1_modelslim_expert_dequant.h"
#endif

template <bool FullPrecisionChannel = false, unsigned Columns = 256, bool FuseNextNorm = false>
__aicore__ inline void ModelSlimExpertCombine(__gm__ int64_t* args) {
  using Integers = pto::Tile<pto::TileType::Vec, int32_t, 1, Columns>;
  using Row = pto::Tile<pto::TileType::Vec, float, 1, Columns>;
  using Bf16 = pto::Tile<pto::TileType::Vec, A1ModelFloat, 1, Columns>;
  using AccGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, Columns>, pto::Stride<1, 1, 1, 2048, 1>>;
  using ScaleGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, Columns>, pto::Stride<1, 1, 1, 2048, 1>>;
  using Bf16Global = pto::GlobalTensor<A1ModelFloat, pto::Shape<1, 1, 1, 1, Columns>, pto::Stride<1, 1, 1, 2048, 1>>;

  static_assert(!FuseNextNorm || (FullPrecisionChannel && Columns == 2048));
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  const unsigned col = FuseNextNorm ? 0 : block * 256;
  auto* acc = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* weightScale = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* routeWeights = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* expertOutput = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[6]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[7]);
  const unsigned tokens = ids->shapes[0];
  if ((tokens != 1 && tokens != 20) || (FuseNextNorm && tokens != 20) || statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + block * 16;
  if (acc->ndims != 2 || acc->shapes[0] != tokens * 8 || acc->shapes[1] != 2048 || inputScale->ndims != 2 ||
      inputScale->shapes[0] != tokens * 8 || inputScale->shapes[1] != 16 || ids->ndims != 2 || ids->shapes[0] != tokens ||
      ids->shapes[1] != 8 || weightScale->ndims != 2 || weightScale->shapes[0] != 128 ||
      weightScale->shapes[1] != 2048 || routeWeights->ndims != 2 || routeWeights->shapes[0] != tokens ||
      routeWeights->shapes[1] != 8 || expertOutput->ndims != 2 || expertOutput->shapes[0] != tokens * 8 ||
      expertOutput->shapes[1] != 2048 || output->ndims != 2 || output->shapes[0] != tokens ||
      output->shapes[1] != 2048) {
    PublishStatus(status, 1);
    return;
  }
  if constexpr (FuseNextNorm) {
    for (unsigned index = 8; index < 13; ++index) {
      auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
      const unsigned expectedRows = index == 9 || index == 10 ? 1 : tokens;
      if (tensor->ndims != 2 || tensor->shapes[0] != expectedRows || tensor->shapes[1] != 2048) {
        PublishStatus(status, 3);
        return;
      }
    }
  }
  for (unsigned route = 0; route < tokens * 8; ++route) {
    const int expert = Data<int32_t>(ids)[route];
    if (expert < 0 || expert >= 128) {
      PublishStatus(status, 2);
      return;
    }
  }
  Integers integer;
  Row values, scales, sum;
  Bf16 rounded;
  pto::TASSIGN<0>(integer);
  pto::TASSIGN<Columns * 4>(values);
  pto::TASSIGN<Columns * 8>(scales);
  pto::TASSIGN<Columns * 12>(sum);
  pto::TASSIGN<Columns * 16>(rounded);
  for (unsigned token = FuseNextNorm ? block : 0; token < tokens; token += FuseNextNorm ? 8 : 1) {
  pto::TEXPANDS(sum, 0.0f);
  pipe_barrier(PIPE_V);
  for (unsigned route = token * 8; route < (token + 1) * 8; ++route) {
    const unsigned expert = Data<int32_t>(ids)[route];
    const float activationScale = Data<float>(inputScale)[route * 16];
    const float routeWeight = Data<float>(routeWeights)[route];
    AccGlobal input(Data<int32_t>(acc) + route * 2048 + col);
    ScaleGlobal channel(Data<float>(weightScale) + expert * 2048 + col);
    pto::TLOAD(integer, input);
    pto::TLOAD(scales, channel);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#ifdef PYPTO_A1_MODEL_FP16
    if constexpr (!FullPrecisionChannel) {
      ModelSlimChannelDequant(integer, scales, rounded);
      pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    } else {
      // Native B20 W2 retains FP32 through channel and token scaling.
      pto::TCVT(values, integer, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TMUL(values, values, scales);
    }
#else
    pto::TCVT(values, integer, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(values, values, scales);
#endif
    pipe_barrier(PIPE_V);
    pto::TMULS(values, values, activationScale);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Bf16Global expertResult(Data<A1ModelFloat>(expertOutput) + route * 2048 + col);
    pto::TSTORE(expertResult, rounded);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMULS(values, values, routeWeight);
    pipe_barrier(PIPE_V);
    pto::TADD(sum, sum, values);
    pipe_barrier(PIPE_ALL);
  }
  pto::TCVT(rounded, sum, pto::RoundMode::CAST_RINT);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  Bf16Global destination(Data<A1ModelFloat>(output) + token * 2048 + col);
  pto::TSTORE(destination, rounded);
  pipe_barrier(PIPE_ALL);
  if constexpr (FuseNextNorm) {
    auto* skip = reinterpret_cast<__gm__ Tensor*>(args[8]);
    auto* gamma = reinterpret_cast<__gm__ Tensor*>(args[9]);
    auto* beta = reinterpret_cast<__gm__ Tensor*>(args[10]);
    auto* normalizedOutput = reinterpret_cast<__gm__ Tensor*>(args[11]);
    auto* skipOutput = reinterpret_cast<__gm__ Tensor*>(args[12]);
    Row skipFloat, square;
    Bf16 skipHalf, gammaHalf, betaHalf;
    pto::TASSIGN<0>(skipFloat);
    pto::TASSIGN<36864>(skipHalf);
    pto::TASSIGN<40960>(gammaHalf);
    pto::TASSIGN<45056>(betaHalf);
    pto::TASSIGN<57344>(square);
    Bf16Global skipInput(Data<A1ModelFloat>(skip) + token * 2048);
    pto::TLOAD(skipHalf, skipInput);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(sum, rounded, pto::RoundMode::CAST_NONE);
    pto::TCVT(skipFloat, skipHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TADD(sum, sum, skipFloat);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, sum, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Bf16Global skipDestination(Data<A1ModelFloat>(skipOutput) + token * 2048);
    pto::TSTORE(skipDestination, rounded);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::TCVT(sum, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(square, sum, sum);
    pipe_barrier(PIPE_V);
    pto::TMULS(square, square, 1.0f / 2048.0f);
    pipe_barrier(PIPE_V);
    using Partial = pto::Tile<pto::TileType::Vec, float, 1, 64>;
    using Scalar = pto::Tile<pto::TileType::Vec, float, 1, 8>;
    Partial partial;
    Scalar variance, root, ones, inverse;
    pto::TASSIGN<49152>(partial);
    pto::TASSIGN<49408>(variance);
    pto::TASSIGN<49440>(root);
    pto::TASSIGN<49472>(ones);
    pto::TASSIGN<49504>(inverse);
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vadds(partial.data(), square.data(), 0.0f, 1, 1, 1, 8, 8);
    for (unsigned column = 64; column < 2048; column += 64) {
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
    pto::TMULS(values, sum, factor);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    Bf16Global gammaInput(Data<A1ModelFloat>(gamma));
    Bf16Global betaInput(Data<A1ModelFloat>(beta));
    pto::TLOAD(gammaHalf, gammaInput);
    pto::TLOAD(betaHalf, betaInput);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(scales, gammaHalf, pto::RoundMode::CAST_NONE);
    pto::TCVT(skipFloat, betaHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(values, values, scales);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TADD(values, values, skipFloat);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Bf16Global nextInput(Data<A1ModelFloat>(normalizedOutput) + token * 2048);
    pto::TSTORE(nextInput, rounded);
    pipe_barrier(PIPE_ALL);
  }
  }
  PublishStatus(status, 0);
}
