/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Row = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Sums = pto::Tile<pto::TileType::Vec, float, 1, 8>;
using Output = pto::Tile<pto::TileType::Vec, A1ModelFloat, 1, 128>;
using InputGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using SumGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;
using OutputGlobal = pto::GlobalTensor<A1ModelFloat, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned head = get_block_idx();
  if (head >= 8) return;
  auto* numerator = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* denominator = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[2]);
  if (numerator->ndims != 2 || numerator->shapes[0] != 8 || numerator->shapes[1] != 128 ||
      denominator->ndims != 2 || denominator->shapes[0] != 1 || denominator->shapes[1] != 8 ||
      output->ndims != 2 || output->shapes[0] != 8 || output->shapes[1] != 128) {
    trap();
    return;
  }
  Row values, divisor, reciprocal, correction;
  Sums sums;
  Output rounded;
  pto::TASSIGN<0>(values);
  pto::TASSIGN<512>(divisor);
  pto::TASSIGN<1024>(reciprocal);
  pto::TASSIGN<1536>(correction);
  pto::TASSIGN<2048>(sums);
  pto::TASSIGN<2080>(rounded);
  InputGlobal source(Data<float>(numerator) + head * 128);
  SumGlobal sumSource(Data<float>(denominator));
  pto::TLOAD(values, source);
  pto::TLOAD(sums, sumSource);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  const float scalar = sums.data()[head];
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pto::TEXPANDS(divisor, scalar);
  pipe_barrier(PIPE_V);
#ifdef PYPTO_A1_MODEL_FP16
  Output halfDivisor;
  pto::TASSIGN<2336>(halfDivisor);
  pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfDivisor, divisor, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(UINT64_MAX, UINT64_MAX);
  vdiv(rounded.data(), rounded.data(), halfDivisor.data(), 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
#else
  set_mask_norm();
  set_vector_mask(0, UINT64_MAX);
  vrec(reciprocal.data(), divisor.data(), 2, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    pto::TMUL(correction, divisor, reciprocal);
    pipe_barrier(PIPE_V);
    pto::TMULS(correction, correction, -1.0f);
    pipe_barrier(PIPE_V);
    pto::TADDS(correction, correction, 2.0f);
    pipe_barrier(PIPE_V);
    pto::TMUL(reciprocal, reciprocal, correction);
    pipe_barrier(PIPE_V);
  }
  pto::TMUL(values, values, reciprocal);
  pipe_barrier(PIPE_V);
  pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
#endif
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  OutputGlobal target(Data<A1ModelFloat>(output) + head * 128);
  pto::TSTORE(target, rounded);
  pipe_barrier(PIPE_ALL);
}
