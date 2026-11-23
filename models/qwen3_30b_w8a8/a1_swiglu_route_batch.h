/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_bounded_int8_pack.h"

template <unsigned Rows>
__aicore__ inline void ModelSlimSwiGLURouteBatch(__gm__ int64_t* args,
                                               unsigned first) {
  constexpr unsigned Cols = 768;
  constexpr unsigned FloatBytes = Rows * Cols * sizeof(float);
  constexpr unsigned HalfBase = 4 * FloatBytes;
  constexpr unsigned QuantBase = HalfBase + Rows * Cols * sizeof(half);
  constexpr unsigned MaximumBase = QuantBase + Rows * Cols;
  constexpr unsigned CarrierBase = MaximumBase + Rows * 128 * sizeof(float);
  constexpr unsigned CarrierBytes = Rows * 8 * sizeof(float);
  static_assert(CarrierBase + 2 * CarrierBytes <= 128 * 1024);
  using Vector = pto::Tile<pto::TileType::Vec, float, Rows, Cols>;
  using Half = pto::Tile<pto::TileType::Vec, half, Rows, Cols>;
  using Integer = pto::Tile<pto::TileType::Vec, int32_t, Rows, Cols>;
  using Row = pto::Tile<pto::TileType::Vec, float, 1, Cols>;
  using HalfInput = pto::GlobalTensor<half, pto::Shape<1, 1, 1, Rows, Cols>,
                                    pto::Stride<1, 1, 1, 8 * 1536, 1>>;
  auto* input = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[6]);
  Vector gate, up, denominator, activated;
  Half halfValues;
  pto::TASSIGN<0>(gate);
  pto::TASSIGN<FloatBytes>(up);
  pto::TASSIGN<2 * FloatBytes>(denominator);
  pto::TASSIGN<3 * FloatBytes>(activated);
  pto::TASSIGN<HalfBase>(halfValues);
  // Preserve the first half of each 64-byte carrier, including its padding.
  // The second half is never touched. Each core owns distinct route rows.
  using ScaleCarrier = pto::Tile<pto::TileType::Vec, float, Rows, 8>;
  using StatusCarrier = pto::Tile<pto::TileType::Vec, int32_t, Rows, 8>;
  using ScaleGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, Rows, 8>,
                                      pto::Stride<1, 1, 1, 8 * 16, 1>>;
  using StatusGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, Rows, 8>,
                                       pto::Stride<1, 1, 1, 8 * 16, 1>>;
  ScaleCarrier scaleCarrier;
  StatusCarrier statusCarrier;
  pto::TASSIGN<CarrierBase>(scaleCarrier);
  pto::TASSIGN<CarrierBase + CarrierBytes>(statusCarrier);
  ScaleGlobal scaleDestination(Data<float>(scale) + first * 16);
  StatusGlobal statusDestination(Data<int32_t>(status) + first * 16);
  pto::TLOAD(scaleCarrier, scaleDestination);
  pto::TLOAD(statusCarrier, statusDestination);
  HalfInput gateInput(Data<half>(input) + first * 1536);
  pto::TLOAD(halfValues, gateInput);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(gate, halfValues, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  HalfInput upInput(Data<half>(input) + first * 1536 + Cols);
  pto::TLOAD(halfValues, upInput);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(up, halfValues, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  for (unsigned row = 0; row < Rows; ++row) {
    const float value = Data<float>(inputScale)[(first / 8 + row) * 16];
    Row gateRow, upRow;
    pto::TASSIGN(gateRow, row * Cols * sizeof(float));
    pto::TASSIGN(upRow, FloatBytes + row * Cols * sizeof(float));
    pto::TMULS(gateRow, gateRow, value);
    pto::TMULS(upRow, upRow, value);
  }
  pipe_barrier(PIPE_V);
  pto::TCVT(halfValues, gate, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(gate, halfValues, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfValues, up, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(up, halfValues, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TMULS(denominator, gate, -1.0f);
  pipe_barrier(PIPE_V);
  pto::TEXP(denominator, denominator);
  pipe_barrier(PIPE_V);
  pto::TADDS(denominator, denominator, 1.0f);
  pipe_barrier(PIPE_V);
  pto::TDIV(activated, gate, denominator);
  pipe_barrier(PIPE_V);
  pto::TMUL(activated, activated, up);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfValues, activated, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(activated, halfValues, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TABS(denominator, activated);
  pipe_barrier(PIPE_V);
  pto::Tile<pto::TileType::Vec, float, 1, Rows * 128> maximumLanes;
  pto::TASSIGN<MaximumBase>(maximumLanes);
  pto::TEXPANDS(maximumLanes, 0.0f);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(0, UINT64_MAX);
  for (unsigned chunk = 0; chunk < Cols; chunk += 128) {
    vmax(maximumLanes.data(), maximumLanes.data(), denominator.data() + chunk,
         Rows, 1, 1, 1, 16, 16, 96);
    vmax(maximumLanes.data() + 64, maximumLanes.data() + 64,
         denominator.data() + chunk + 64, Rows, 1, 1, 1, 16, 16, 96);
    pipe_barrier(PIPE_V);
  }
  for (unsigned row = 0; row < Rows; ++row) {
    vmax(gate.data() + row * Cols, maximumLanes.data() + row * 128,
         maximumLanes.data() + row * 128 + 64, 1, 1, 1, 1, 8, 8, 8);
  }
  pipe_barrier(PIPE_V);
  for (unsigned row = 0; row < Rows; ++row) {
    vcmax(up.data() + row * Cols, gate.data() + row * Cols,
          1, 1, 1, 8, false, false, false);
  }
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  float inverses[Rows];
  for (unsigned row = 0; row < Rows; ++row) {
    const float storedScale = up.data()[row * Cols] / 127.0f;
    scaleCarrier.data()[row * 8] = storedScale;
    inverses[row] = storedScale > 1e-7f ? 1.0f / storedScale : 0.0f;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  for (unsigned row = 0; row < Rows; ++row) {
    Row activationRow;
    pto::TASSIGN(activationRow, 3 * FloatBytes + row * Cols * sizeof(float));
    pto::TMULS(activationRow, activationRow, inverses[row]);
  }
  pipe_barrier(PIPE_V);
  Integer integers;
  pto::TASSIGN<0>(integers);
  pto::TCVT(integers, activated, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(up, integers, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfValues, up, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::Tile<pto::TileType::Vec, half, 1, Rows * Cols> flatHalf;
  pto::Tile<pto::TileType::Vec, int8_t, 1, Rows * Cols> flatQuant;
  pto::TASSIGN<HalfBase>(flatHalf);
  pto::TASSIGN<QuantBase>(flatQuant);
  PackBoundedIntegralInt8(flatQuant, flatHalf);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  using Quant = pto::Tile<pto::TileType::Vec, int8_t, Rows, Cols>;
  using Destination = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, Rows, Cols>,
                                      pto::Stride<1, 1, 1, 8 * Cols, 1>>;
  Quant output;
  pto::TASSIGN<QuantBase>(output);
  Destination destination(Data<int8_t>(quant) + first * Cols);
  pto::TSTORE(destination, output);
  pipe_barrier(PIPE_ALL);
  for (unsigned row = 0; row < Rows; ++row) {
    statusCarrier.data()[row * 8] = 0;
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
  pto::TSTORE(scaleDestination, scaleCarrier);
  pto::TSTORE(statusDestination, statusCarrier);
  pipe_barrier(PIPE_ALL);
}

__aicore__ inline bool TryModelSlimSwiGLURouteGroups(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  if (ids->shapes[0] != 20 || inputScale->shapes[0] != 20) return false;
  for (unsigned route = block; route < 160; route += 8) {
    const int32_t expert = Data<int32_t>(ids)[route];
    if (expert < 0 || expert >= 128) return false;
  }
  for (unsigned token = 0; token < 20; token += 4) {
    ModelSlimSwiGLURouteBatch<4>(args, block + token * 8);
  }
  return true;
}
