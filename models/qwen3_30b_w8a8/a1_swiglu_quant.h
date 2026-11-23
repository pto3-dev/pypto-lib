/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"
#ifdef PYPTO_A1_MODEL_FP16
#include "a1_bounded_int8_pack.h"
#include "a1_swiglu_route_batch.h"
#include "a1_modelslim_expert_dequant.h"
#include "a1_modelslim_expert_dequant_packed.h"
#endif

template <bool NativeBatchedScale = false, bool RetainActivated = false, bool VectorMaximum = false, bool PackedDequant = false, bool ChannelDequantizedInput = false>
__aicore__ inline void ModelSlimSwiGLUQuant(__gm__ int64_t* args) {
  static_assert(!ChannelDequantizedInput || (NativeBatchedScale && PackedDequant));
#ifndef PYPTO_A1_MODEL_FP16
  static_assert(!ChannelDequantizedInput, "Grouped W13 dequantization requires FP16");
#endif
  using AccGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 1536, 1>>;
  using WeightGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 1536, 1>>;
  constexpr unsigned VectorCols = ChannelDequantizedInput ? 768 : 128;
  constexpr unsigned BufferScale = VectorCols / 128;
  using Acc = pto::Tile<pto::TileType::Vec, int32_t, 1, VectorCols>;
  using Row = pto::Tile<pto::TileType::Vec, float, 1, VectorCols>;
  using Half = pto::Tile<pto::TileType::Vec, half, 1, VectorCols>;
  using Quant = pto::Tile<pto::TileType::Vec, int8_t, 1, VectorCols>;
  using QuantGlobal = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, VectorCols>,
                                       pto::Stride<1, 1, 1, 768, 1>>;
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* acc = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* inputScale = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* weightScale = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* quant = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* scale = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
  const unsigned routes = ids->shapes[0] * 8;
  if ((routes != 8 && routes != 160) || statusTensor->ndims != 2 || statusTensor->shapes[0] != routes || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + block * 16;
  if (acc->ndims != 2 || acc->shapes[0] != routes || acc->shapes[1] != 1536 || inputScale->ndims != 2 ||
      (inputScale->shapes[0] != routes && inputScale->shapes[0] != routes / 8) || inputScale->shapes[1] != 16 || ids->ndims != 2 || ids->shapes[0] != routes / 8 ||
      ids->shapes[1] != 8 || weightScale->ndims != 2 || weightScale->shapes[0] != 128 ||
      weightScale->shapes[1] != (PackedDequant ? 3072 : 1536) || quant->ndims != 2 || quant->shapes[0] != routes ||
      quant->shapes[1] != 768 || scale->ndims != 2 || scale->shapes[0] != routes || scale->shapes[1] != 16) {
    PublishStatus(status, 1);
    return;
  }
#ifdef PYPTO_A1_MODEL_FP16
  if constexpr (NativeBatchedScale && RetainActivated && VectorMaximum && PackedDequant &&
                ChannelDequantizedInput) {
    if (TryModelSlimSwiGLURouteGroups(args)) return;
  }
#endif
  for (unsigned route = block; route < routes; route += 8) {
  status = Data<int32_t>(statusTensor) + route * 16;
  const int32_t expert = Data<int32_t>(ids)[route];
  if (expert < 0 || expert >= 128) {
    PublishStatus(status, 2);
    return;
  }
  const float activationScale = Data<float>(inputScale)[(inputScale->shapes[0] == routes ? route : route / 8) * 16];
  Acc gateAcc, upAcc;
  Row gate, up, gateScale, upScale, denominator, activated;
  Half halfValues;
  Quant output;
  pto::TASSIGN<0 * BufferScale>(gateAcc);
  pto::TASSIGN<512 * BufferScale>(upAcc);
  pto::TASSIGN<1024 * BufferScale>(gate);
  pto::TASSIGN<1536 * BufferScale>(up);
  pto::TASSIGN<2048 * BufferScale>(gateScale);
  pto::TASSIGN<2560 * BufferScale>(upScale);
  pto::TASSIGN<3072 * BufferScale>(denominator);
  pto::TASSIGN<3584 * BufferScale>(activated);
  pto::TASSIGN<4096 * BufferScale>(halfValues);
  pto::TASSIGN<4352 * BufferScale>(output);
  pto::Tile<pto::TileType::Vec, float, 1, 128> maximumLanes;
  pto::TASSIGN<VectorCols == 128 ? 18432 : 28672>(maximumLanes);
  if constexpr (VectorMaximum) {
    pto::TEXPANDS(maximumLanes, 0.0f);
    pipe_barrier(PIPE_V);
  }
  using Packed = pto::Tile<pto::TileType::Vec, int32_t, 1, 256>;
  using PackedGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 256>,
                                          pto::Stride<1, 1, 1, 3072, 1>>;
  Packed gatePacked, upPacked;
  pto::TASSIGN<20480>(gatePacked);
  pto::TASSIGN<21504>(upPacked);
  float maximum = 0.0f;
  float inverseScale = 0.0f;
  // Keep the FP16-rounded activation in UB, away from dequant scratch8192..9856.
  // Retain the rounded activation between amax and quantization passes.
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (unsigned chunk = 0; chunk < 768; chunk += VectorCols) {
      Row retained;
      pto::TASSIGN(retained, (VectorCols == 128 ? 12288 : 32768) + chunk * sizeof(float));
      if (RetainActivated && pass == 1) {
        pto::TMULS(activated, retained, 1.0f);
        pipe_barrier(PIPE_V);
      } else {
      if constexpr (ChannelDequantizedInput) {
        using HalfInput = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, VectorCols>,
                                            pto::Stride<1, 1, 1, 1536, 1>>;
        Half gateHalf, upHalf;
        pto::TASSIGN<VectorCols == 128 ? 8192 : 36352>(gateHalf);
        pto::TASSIGN<VectorCols == 128 ? 8448 : 37888>(upHalf);
        HalfInput gateInput(Data<half>(acc) + route * 1536 + chunk);
        HalfInput upInput(Data<half>(acc) + route * 1536 + 768 + chunk);
        pto::TLOAD(gateHalf, gateInput);
        pto::TLOAD(upHalf, upInput);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TCVT(gate, gateHalf, pto::RoundMode::CAST_NONE);
        pto::TCVT(up, upHalf, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
      } else {
      AccGlobal gateInput(Data<int32_t>(acc) + route * 1536 + chunk);
      AccGlobal upInput(Data<int32_t>(acc) + route * 1536 + 768 + chunk);
      WeightGlobal gateWeights(Data<float>(weightScale) + expert * 1536 + chunk);
      WeightGlobal upWeights(Data<float>(weightScale) + expert * 1536 + 768 + chunk);
      pto::TLOAD(gateAcc, gateInput);
      pto::TLOAD(upAcc, upInput);
      if constexpr (PackedDequant) {
        PackedGlobal gateTable(Data<int32_t>(weightScale) + expert * 3072 + chunk * 2);
        PackedGlobal upTable(Data<int32_t>(weightScale) + expert * 3072 + 1536 + chunk * 2);
        pto::TLOAD(gatePacked, gateTable);
        pto::TLOAD(upPacked, upTable);
      } else {
      pto::TLOAD(gateScale, gateWeights);
      pto::TLOAD(upScale, upWeights);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pto::TCVT(gate, gateAcc, pto::RoundMode::CAST_NONE);
      pto::TCVT(up, upAcc, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      }
#ifdef PYPTO_A1_MODEL_FP16
      if constexpr (!ChannelDequantizedInput) {
      if constexpr (PackedDequant) PackedModelSlimChannelDequant128(gateAcc, gatePacked, halfValues);
      else ModelSlimChannelDequant(gateAcc, gateScale, halfValues);
      pto::TCVT(gate, halfValues, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      if constexpr (PackedDequant) PackedModelSlimChannelDequant128(upAcc, upPacked, halfValues);
      else ModelSlimChannelDequant(upAcc, upScale, halfValues);
      pto::TCVT(up, halfValues, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      }
      pto::TMULS(gate, gate, activationScale);
      pto::TMULS(up, up, activationScale);
      pipe_barrier(PIPE_V);
      pto::TCVT(halfValues, gate, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(gate, halfValues, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TCVT(halfValues, up, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(up, halfValues, pto::RoundMode::CAST_NONE);
#else
      pto::TMULS(gate, gate, activationScale);
      pto::TMULS(up, up, activationScale);
      pipe_barrier(PIPE_V);
      pto::TMUL(gate, gate, gateScale);
      pto::TMUL(up, up, upScale);
#endif
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
#ifdef PYPTO_A1_MODEL_FP16
      pto::TCVT(halfValues, activated, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(activated, halfValues, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
#endif
      }
      if (pass == 0) {
        if constexpr (RetainActivated) {
          pto::TMULS(retained, activated, 1.0f);
          pipe_barrier(PIPE_V);
        }
        if constexpr (VectorMaximum) {
          pto::TABS(denominator, activated);
          pipe_barrier(PIPE_V);
          set_mask_norm();
          set_vector_mask(0, UINT64_MAX);
          for (unsigned group = 0; group < VectorCols; group += 128) {
            vmax(maximumLanes.data(), maximumLanes.data(), denominator.data() + group,
                 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
          }
        } else {
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          for (unsigned lane = 0; lane < 128; ++lane) {
            const float value = activated.data()[lane];
            const float magnitude = value < 0.0f ? -value : value;
            if (magnitude > maximum) maximum = magnitude;
          }
          set_flag(PIPE_S, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }
      } else {
        pto::TMULS(activated, activated, inverseScale);
        pipe_barrier(PIPE_V);
#ifdef PYPTO_A1_MODEL_FP16
        // QuantGMM rounds scaled FP32 to INT32 before its exact FP16 hop.
        pto::TCVT(gateAcc, activated, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(gate, gateAcc, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        pto::TCVT(halfValues, gate, pto::RoundMode::CAST_RINT);
#else
        // This FP16 rounding is the checkpoint's INT8 epilogue contract, not
        // conversion of model weights, hidden states or attention to FP16.
        pto::TCVT(halfValues, activated, pto::RoundMode::CAST_RINT);
#endif
        pipe_barrier(PIPE_V);
#ifdef PYPTO_A1_MODEL_FP16
        PackBoundedIntegralInt8(output, halfValues);
#else
        pto::TCVT(output, halfValues, pto::RoundMode::CAST_RINT);
#endif
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        QuantGlobal destination(Data<int8_t>(quant) + route * 768 + chunk);
        pto::TSTORE(destination, output);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    if (pass == 0) {
      if constexpr (VectorMaximum) {
        set_mask_norm();
        set_vector_mask(0, UINT64_MAX);
        vmax(denominator.data(), maximumLanes.data(), maximumLanes.data() + 64,
             1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vcmax(gate.data(), denominator.data(), 1, 1, 1, 8, false, false, false);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        maximum = gate.data()[0];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      }
#ifdef PYPTO_A1_MODEL_FP16
      // Native B20 GMM divides amax by 127 rather than multiplying by its
      // rounded FP32 reciprocal; the difference is observable at half midpoints.
      const float storedScale = NativeBatchedScale ? maximum / 127.0f : maximum * (1.0f / 127.0f);
#else
      const float storedScale = maximum / 127.0f;
#endif
      auto* scaleOutput = Data<float>(scale) + route * 16;
      scaleOutput[0] = storedScale;
      dcci((__gm__ void*)scaleOutput, cache_line_t::SINGLE_CACHE_LINE);
#ifdef PYPTO_A1_MODEL_FP16
      inverseScale = storedScale > 1e-7f ? 1.0f / storedScale : 0.0f;
#else
      inverseScale = maximum > 0.0f ? 1.0f / storedScale : 0.0f;
#endif
    }
  }
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
  }
}
