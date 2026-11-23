/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

using HalfRow = pto::Tile<pto::TileType::Vec, half, 1, 128>;
using Row = pto::Tile<pto::TileType::Vec, float, 1, 128>;
using Rows = pto::Tile<pto::TileType::Vec, float, 8, 128>;
using Column = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor>;
using Selected = pto::Tile<pto::TileType::Vec, float, 1, 16>;
using SelectedHalf = pto::Tile<pto::TileType::Vec, half, 1, 16>;
using Weights = pto::Tile<pto::TileType::Vec, float, 1, 8>;
using Ids = pto::Tile<pto::TileType::Vec, int32_t, 1, 8>;
using InputGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using WeightGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;
using IdGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;

template <bool FusedNativeRouting = false>
__aicore__ inline void ModelSlimTopK(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* logits = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* weights = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[3]);
  const unsigned tokens = logits->shapes[0];
  if ((tokens != 1 && tokens != 20) || logits->ndims != 2 || logits->shapes[0] != tokens || logits->shapes[1] != 128 ||
      ids->ndims != 2 || ids->shapes[0] != tokens || ids->shapes[1] != 8 ||
      weights->ndims != 2 || weights->shapes[0] != tokens || weights->shapes[1] != 8 ||
      status->ndims != 2 || status->shapes[0] != tokens || status->shapes[1] != 16) {
    trap();
    return;
  }
  HalfRow input, halfProbability;
  Row row, centered, exponential, probability;
  Rows rows, scratch;
  Column reduced;
  Selected selected, divisor, expanded;
  SelectedHalf halfSelected, halfDivisor;
  Weights output;
  Ids indices;
  pto::TASSIGN<0>(input);
  pto::TASSIGN<256>(row);
  pto::TASSIGN<768>(centered);
  pto::TASSIGN<1280>(exponential);
  pto::TASSIGN<1792>(probability);
  pto::TASSIGN<2816>(halfProbability);
  pto::TASSIGN<3072>(rows);
  pto::TASSIGN<7168>(scratch);
  pto::TASSIGN<11264>(reduced);
  pto::TASSIGN<11296>(indices);
  pto::TASSIGN<11328>(selected);
  pto::TASSIGN<11392>(divisor);
  pto::TASSIGN<11456>(halfSelected);
  pto::TASSIGN<11488>(halfDivisor);
  pto::TASSIGN<11520>(expanded);
  pto::TASSIGN<11520>(output);
  for (unsigned token = block; token < tokens; token += 8) {
  InputGlobal source(Data<half>(logits) + token * 128);
  pto::TLOAD(input, source);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(row, input, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TCOLEXPAND(rows, row);
  pipe_barrier(PIPE_V);
  pto::TROWMAX(reduced, rows, scratch);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float maximum = reduced.data()[0];
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pto::TADDS(centered, row, -maximum);
  pipe_barrier(PIPE_V);
  // Separate FP16 Softmax narrows x-max; the fused 310P routing op does not.
  if constexpr (!FusedNativeRouting) {
    pto::TCVT(halfProbability, centered, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    pto::TCVT(centered, halfProbability, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
  }
  pto::TEXP(exponential, centered);
  pipe_barrier(PIPE_V);
  pto::TCOLEXPAND(rows, exponential);
  pipe_barrier(PIPE_V);
  pto::TROWSUM(reduced, rows, scratch);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float total = reduced.data()[0];
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // Preserve reciprocal-then-multiply rounding before the FP16 Softmax
  // materialization. Direct division differs at representable half midpoints.
  pto::TMULS(probability, exponential, 1.0f / total);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfProbability, probability, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(probability, halfProbability, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  float selectedSum = 0.0f;
  // Keep probabilities intact for normalization; mask winners in a private row.
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pto::TMULS(centered, probability, 1.0f);
  pipe_barrier(PIPE_V);
  for (unsigned slot = 0; slot < 8; ++slot) {
    set_mask_norm();
    set_vector_mask(0, UINT64_MAX);
    vcmax(row.data(), centered.data(), 1, 1, 1, 8, false, false, false);
    vcmax(row.data() + 8, centered.data() + 64, 1, 1, 1, 8, false, false, false);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const auto* positions = reinterpret_cast<__ubuf__ uint32_t*>(row.data());
    const unsigned half = row.data()[8] > row.data()[0] ? 1 : 0;
    const unsigned bestIndex = positions[half * 8 + 1] + half * 64;
    indices.data()[slot] = bestIndex;
    selected.data()[slot] = probability.data()[bestIndex];
    selectedSum += probability.data()[bestIndex];
    centered.data()[bestIndex] = -1.0f;
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (unsigned slot = 0; slot < 16; ++slot) {
    if (slot >= 8) selected.data()[slot] = 0.0f;
    divisor.data()[slot] = selectedSum;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  pto::TCVT(halfSelected, selected, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  pto::TCVT(halfDivisor, divisor, pto::RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(0, 0xffff);
  vdiv(halfSelected.data(), halfSelected.data(), halfDivisor.data(), 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  pto::TCVT(expanded, halfSelected, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  WeightGlobal weightOutput(Data<float>(weights) + token * 8);
  IdGlobal idOutput(Data<int32_t>(ids) + token * 8);
  pto::TSTORE(weightOutput, output);
  pto::TSTORE(idOutput, indices);
  pipe_barrier(PIPE_ALL);
  PublishStatus(Data<int32_t>(status) + token * 16, 0);
  }
}
