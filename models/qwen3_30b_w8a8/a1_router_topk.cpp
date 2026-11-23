/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Input = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 128>;
using Row = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Rows = pto::Tile<pto::TileType::Vec, float, 8, 128>;
using Column = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor>;
using Selected = pto::Tile<pto::TileType::Vec, float, 1, 8>;
using SelectedRows = pto::Tile<pto::TileType::Vec, float, 8, 8>;
using Ids = pto::Tile<pto::TileType::Vec, int32_t, 1, 8>;
using InputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using WeightGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;
using IdGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  if (get_block_idx() != 0) return;
  auto* logits = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* weights = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 1 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor);
  if (logits->ndims != 2 || logits->shapes[0] != 1 || logits->shapes[1] != 128 || ids->ndims != 2 ||
      ids->shapes[0] != 1 || ids->shapes[1] != 8 || weights->ndims != 2 || weights->shapes[0] != 1 ||
      weights->shapes[1] != 8) {
    PublishStatus(status, 1);
    return;
  }
  Input input;
  Row row;
  Rows rows, expanded, exponential, probability, scratch;
  Column maximum, total, selectedTotal;
  Selected selected, normalized;
  SelectedRows selectedRows, selectedScratch, selectedDenominator, selectedNormalized;
  Ids indices;
  pto::TASSIGN<0>(input);
  pto::TASSIGN<256>(row);
  pto::TASSIGN<768>(rows);
  pto::TASSIGN<4864>(expanded);
  pto::TASSIGN<8960>(exponential);
  pto::TASSIGN<13056>(probability);
  pto::TASSIGN<17152>(scratch);
  pto::TASSIGN<21248>(maximum);
  pto::TASSIGN<21280>(total);
  pto::TASSIGN<21312>(selectedTotal);
  pto::TASSIGN<21344>(selected);
  pto::TASSIGN<21376>(normalized);
  pto::TASSIGN<21408>(selectedRows);
  pto::TASSIGN<21664>(selectedScratch);
  pto::TASSIGN<21920>(selectedDenominator);
  pto::TASSIGN<22176>(selectedNormalized);
  pto::TASSIGN<22432>(indices);
  InputGlobal inputGlobal(Data<bfloat16_t>(logits));
  pto::TLOAD(input, inputGlobal);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(row, input, pto::RoundMode::CAST_NONE);
  pto::TCOLEXPAND(rows, row);
  pipe_barrier(PIPE_V);
  pto::TROWMAX(maximum, rows, scratch);
  pto::TROWEXPAND(expanded, maximum);
  pipe_barrier(PIPE_V);
  pto::TSUB(exponential, rows, expanded);
  pipe_barrier(PIPE_V);
  pto::TEXP(exponential, exponential);
  pipe_barrier(PIPE_V);
  pto::TROWSUM(total, exponential, scratch);
  pto::TROWEXPAND(expanded, total);
  pipe_barrier(PIPE_V);
  pto::TDIV(probability, exponential, expanded);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (unsigned slot = 0; slot < 8; ++slot) {
    int bestIndex = -1;
    float best = -1.0f;
    for (unsigned expert = 0; expert < 128; ++expert) {
      bool used = false;
      for (unsigned previous = 0; previous < slot; ++previous)
        used |= indices.data()[previous] == static_cast<int>(expert);
      const float value = probability.data()[expert];
      // Stable lower-index ordering makes equal BF16 logits deterministic.
      if (!used && (bestIndex < 0 || value > best)) {
        best = value;
        bestIndex = expert;
      }
    }
    indices.data()[slot] = bestIndex;
    selected.data()[slot] = best;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pto::TCOLEXPAND(selectedRows, selected);
  pipe_barrier(PIPE_V);
  pto::TROWSUM(selectedTotal, selectedRows, selectedScratch);
  pto::TROWEXPAND(selectedDenominator, selectedTotal);
  pipe_barrier(PIPE_V);
  pto::TDIV(selectedNormalized, selectedRows, selectedDenominator);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  auto* inputBits = reinterpret_cast<__ubuf__ uint32_t*>(selectedNormalized.data());
  auto* outputBits = reinterpret_cast<__ubuf__ uint32_t*>(normalized.data());
  for (unsigned slot = 0; slot < 8; ++slot)
    outputBits[slot] = pto::detail::A1Bf16ToFp32Bits(pto::detail::A1Fp32ToBf16Bits(inputBits[slot]));
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  WeightGlobal weightOutput(Data<float>(weights));
  IdGlobal idOutput(Data<int32_t>(ids));
  pto::TSTORE(weightOutput, normalized);
  pto::TSTORE(idOutput, indices);
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
