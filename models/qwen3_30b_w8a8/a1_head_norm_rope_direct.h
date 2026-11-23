// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#pragma once
#include "a1_tensor.h"

// Same PTO arithmetic and FP16 boundaries as the qualified generated head kernel.
// Runtime-selected phase rows use direct GlobalTensor pointers, with no partition
// of a larger compiler tile and no static unrolling of predicated head bodies.
template <unsigned PaddedHeads>
__aicore__ inline void A1DirectHeadNormRope(__gm__ int64_t* args) {
  static_assert(PaddedHeads == 8 || PaddedHeads == 32);
  set_mask_norm();
  set_vector_mask(-1, -1);
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* rope = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* norm = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* rotated = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (x->ndims != 2 || x->shapes[0] != 20 * PaddedHeads || x->shapes[1] != 128 ||
      weight->ndims != 2 || weight->shapes[0] != 1 || weight->shapes[1] != 128 ||
      rope->ndims != 2 || rope->shapes[0] != 20 || rope->shapes[1] != 128 ||
      norm->ndims != 2 || norm->shapes[0] != x->shapes[0] || norm->shapes[1] != 128 ||
      rotated->ndims != 2 || rotated->shapes[0] != x->shapes[0] || rotated->shapes[1] != 128) {
    trap();
    return;
  }
  using FullHalf = pto::Tile<pto::TileType::Vec, half, 8, 128>;
  using FullFloat = pto::Tile<pto::TileType::Vec, float, 8, 128>;
  using Half = pto::Tile<pto::TileType::Vec, half, 8, 64>;
  using Value = pto::Tile<pto::TileType::Vec, float, 8, 64>;
  using ParameterHalf = pto::Tile<pto::TileType::Vec, half, 1, 64>;
  using Parameter = pto::Tile<pto::TileType::Vec, float, 1, 64>;
  using ScalarRow = pto::Tile<pto::TileType::Vec, float, 1, 8>;
  using ScalarColumn = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor,
                                 8, 1, pto::SLayout::NoneBox, 512>;
  using FullGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 128>,
                                      pto::Stride<1, 1, 1, 128, 1>>;
  using HalfGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 64>,
                                      pto::Stride<1, 1, 1, 128, 1>>;
  using ParameterGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>,
                                           pto::Stride<1, 1, 1, 64, 1>>;
  FullHalf raw;
  FullFloat square, scratch;
  ScalarRow variance, root, ones, inverse;
  ScalarColumn varianceColumn, inverseColumn;
  ParameterHalf gammaHalfL, gammaHalfR, cosHalf, sinHalf;
  Parameter gammaL, gammaR, cosine, sine;
  Value expandedInverse, expandedGammaL, expandedGammaR, expandedCos, expandedSin;
  Half halfL, halfR, productHalf;
  Value left, right, leftCos, rightSin, rightCos, leftSin;
  pto::TASSIGN<0>(raw);
  pto::TASSIGN<2048>(square);
  pto::TASSIGN<6144>(scratch);
  pto::TASSIGN<12288>(variance);
  pto::TASSIGN<12288>(varianceColumn);
  pto::TASSIGN<12800>(root);
  pto::TASSIGN<13312>(ones);
  pto::TASSIGN<13824>(inverse);
  pto::TASSIGN<13824>(inverseColumn);
  pto::TASSIGN<14336>(gammaHalfL);
  pto::TASSIGN<14592>(gammaHalfR);
  pto::TASSIGN<14848>(gammaL);
  pto::TASSIGN<15104>(gammaR);
  pto::TASSIGN<15360>(cosHalf);
  pto::TASSIGN<15616>(sinHalf);
  pto::TASSIGN<15872>(cosine);
  pto::TASSIGN<16128>(sine);
  pto::TASSIGN<16384>(expandedInverse);
  pto::TASSIGN<18432>(expandedGammaL);
  pto::TASSIGN<20480>(expandedGammaR);
  pto::TASSIGN<22528>(expandedCos);
  pto::TASSIGN<24576>(expandedSin);
  pto::TASSIGN<26624>(halfL);
  pto::TASSIGN<27648>(halfR);
  pto::TASSIGN<28672>(left);
  pto::TASSIGN<30720>(right);
  pto::TASSIGN<32768>(leftCos);
  pto::TASSIGN<34816>(rightSin);
  pto::TASSIGN<36864>(rightCos);
  pto::TASSIGN<38912>(leftSin);
  pto::TASSIGN<40960>(productHalf);
  ParameterGlobal gammaSourceL(Data<half>(weight));
  ParameterGlobal gammaSourceR(Data<half>(weight) + 64);
  pto::TLOAD(gammaHalfL, gammaSourceL);
  pto::TLOAD(gammaHalfR, gammaSourceR);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  pto::TCVT(gammaL, gammaHalfL, pto::RoundMode::CAST_NONE);
  pto::TCVT(gammaR, gammaHalfR, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TCOLEXPAND(expandedGammaL, gammaL);
  pto::TCOLEXPAND(expandedGammaR, gammaR);
  pipe_barrier(PIPE_ALL);
  for (unsigned task = block; task < 20 * PaddedHeads / 8; task += 8) {
    const unsigned base = task * 8;
    const unsigned row = base / PaddedHeads;
    FullGlobal input(Data<half>(x) + base * 128);
    ParameterGlobal phaseL(Data<half>(rope) + row * 128);
    ParameterGlobal phaseR(Data<half>(rope) + row * 128 + 64);
    pto::TLOAD(raw, input);
    pto::TLOAD(cosHalf, phaseL);
    pto::TLOAD(sinHalf, phaseR);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TCVT(square, raw, pto::RoundMode::CAST_NONE);
    pto::TCVT(cosine, cosHalf, pto::RoundMode::CAST_NONE);
    pto::TCVT(sine, sinHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(square, square, square);
    pipe_barrier(PIPE_V);
    pto::TROWSUM(varianceColumn, square, scratch);
    pipe_barrier(PIPE_V);
    pto::TMULS(variance, variance, 1.0f / 128.0f);
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
    pipe_barrier(PIPE_ALL);
    // Split the already loaded FP16 input, preserving its exact stored bits.
    copy_ubuf_to_ubuf(halfL.data(), raw.data(), 0, 8, 4, 4, 0);
    copy_ubuf_to_ubuf(halfR.data(), raw.data() + 64, 0, 8, 4, 4, 0);
    pipe_barrier(PIPE_ALL);
    pto::TCVT(left, halfL, pto::RoundMode::CAST_NONE);
    pto::TCVT(right, halfR, pto::RoundMode::CAST_NONE);
    pto::TROWEXPAND(expandedInverse, inverseColumn);
    pto::TCOLEXPAND(expandedCos, cosine);
    pto::TCOLEXPAND(expandedSin, sine);
    pipe_barrier(PIPE_V);
    pto::TMUL(left, left, expandedInverse);
    pto::TMUL(right, right, expandedInverse);
    pipe_barrier(PIPE_V);
    pto::TMUL(left, left, expandedGammaL);
    pto::TMUL(right, right, expandedGammaR);
    pipe_barrier(PIPE_V);
    pto::TCVT(halfL, left, pto::RoundMode::CAST_NONE);
    pto::TCVT(halfR, right, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    HalfGlobal normL(Data<half>(norm) + base * 128);
    HalfGlobal normR(Data<half>(norm) + base * 128 + 64);
    pto::TSTORE(normL, halfL);
    pto::TSTORE(normR, halfR);
    pipe_barrier(PIPE_ALL);
    pto::TCVT(left, halfL, pto::RoundMode::CAST_NONE);
    pto::TCVT(right, halfR, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TMUL(leftCos, left, expandedCos);
    pto::TMUL(rightSin, right, expandedSin);
    pto::TMUL(rightCos, right, expandedCos);
    pto::TMUL(leftSin, left, expandedSin);
    pipe_barrier(PIPE_V);
    pto::TCVT(productHalf, leftCos, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(leftCos, productHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(productHalf, rightSin, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(rightSin, productHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(productHalf, rightCos, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(rightCos, productHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(productHalf, leftSin, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TCVT(leftSin, productHalf, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    pto::TSUB(left, leftCos, rightSin);
    pto::TADD(right, rightCos, leftSin);
    pipe_barrier(PIPE_V);
    pto::TCVT(halfL, left, pto::RoundMode::CAST_NONE);
    pto::TCVT(halfR, right, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    HalfGlobal outputL(Data<half>(rotated) + base * 128);
    HalfGlobal outputR(Data<half>(rotated) + base * 128 + 64);
    pto::TSTORE(outputL, halfL);
    pto::TSTORE(outputR, halfR);
    pipe_barrier(PIPE_ALL);
  }
}
