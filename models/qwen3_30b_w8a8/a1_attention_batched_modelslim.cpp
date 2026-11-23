/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

#if defined(PYPTO_A1_PREFETCH_VALUES) && \
    (!defined(PYPTO_A1_DIRECT_PAGES) || !defined(PYPTO_A1_WIDE_QK) || !defined(PYPTO_A1_WIDE_PV))
#error "A1 V-page prefetch requires direct pages and wide QK/PV"
#endif

#ifdef PYPTO_A1_VECTOR_COPY
__aicore__ inline void CopyHeadColumns(__ubuf__ float* dst, __ubuf__ float* src) {
  // Copy16 active FP32 lanes per row; repeat strides use32-byte units.
  set_mask_norm();
  set_vector_mask(0ULL, 65535ULL);
  vmuls(dst, src, 1.0f, 8, 1, 1, 16, 2);
  pipe_barrier(PIPE_V);
  set_vector_mask(UINT64_MAX, UINT64_MAX);
}
#endif

// Eight-core grid-strided GQA. Each task owns one request and one KV head.
// Keep native page-128 FP16 score/probability and final division boundaries.
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* query = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* keys = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* values = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* lengths = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[5]);
#ifdef PYPTO_A1_DIRECT_PAGES
  auto* table = reinterpret_cast<__gm__ Tensor*>(args[5]);
  status = reinterpret_cast<__gm__ Tensor*>(args[6]);
#endif
  if (query->ndims != 3 || query->shapes[0] != 80 || query->shapes[1] != 16 || query->shapes[2] != 128 ||
#ifdef PYPTO_A1_DIRECT_PAGES
      keys->ndims != 4 || keys->shapes[1] != 32 || keys->shapes[2] != 128 || keys->shapes[3] != 16 ||
      values->ndims != 4 || values->shapes[0] != keys->shapes[0] || values->shapes[1] != 32 ||
      values->shapes[2] != 128 || values->shapes[3] != 16 ||
      table->ndims != 2 || table->shapes[0] != 20 || table->shapes[1] != 32 ||
#else
      keys->ndims != 3 || keys->shapes[0] != 80 || keys->shapes[1] != 4096 || keys->shapes[2] != 128 ||
      values->ndims != 3 || values->shapes[0] != 80 || values->shapes[1] != 4096 || values->shapes[2] != 128 ||
#endif
      lengths->ndims != 2 || lengths->shapes[0] != 20 || lengths->shapes[1] != 8 ||
      output->ndims != 2 || output->shapes[0] != 20 || output->shapes[1] != 4096 ||
      status->ndims != 2 || status->shapes[0] != 80 || status->shapes[1] != 16) {
    trap();
    return;
  }
  using Mat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor,
                        16, 16, pto::SLayout::RowMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::ColMajor,
                         16, 16, pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor,
                          16, 16, pto::SLayout::ColMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor,
                        16, 16, pto::SLayout::RowMajor, 1024>;
  using BlockFloat = pto::Tile<pto::TileType::Vec, float, 16, 16>;
  using BlockHalf = pto::Tile<pto::TileType::Vec, half, 16, 16>;
  using Rows = pto::Tile<pto::TileType::Vec, float, 8, 128>;
  using HalfRows = pto::Tile<pto::TileType::Vec, half, 8, 128>;
  using PaddedHalf = pto::Tile<pto::TileType::Vec, half, 16, 128>;
  using Column = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor>;
  using MatrixGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 128, 1>>;
  using CacheGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 16, 1>>;
  using OutputGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  Mat ma, mb;
  Left left;
#ifdef PYPTO_A1_WIDE_QK
  using QueryStorage = pto::Tile<pto::TileType::Mat, half, 16, 128, pto::BLayout::RowMajor,
                                 16, 128, pto::SLayout::RowMajor, 512>;
  using QueryOperand = pto::Tile<pto::TileType::Left, half, 16, 128, pto::BLayout::ColMajor,
                                 16, 128, pto::SLayout::RowMajor, 512>;
  QueryStorage queryCache;
  QueryOperand queryLeft;
  // L1 [65536,69632) and L0A [8192,12288) are distinct from KV/PV staging.
  pto::TASSIGN<65536>(queryCache);
  pto::TASSIGN<8192>(queryLeft);
#endif
  Right right;
  Acc acc;
#if defined(PYPTO_A1_WIDE_QK) || defined(PYPTO_A1_WIDE_PV)
  using PageMat = pto::Tile<pto::TileType::Mat, half, 128, 128, pto::BLayout::RowMajor,
                            128, 128, pto::SLayout::ColMajor, 512>;
  using WideRight = pto::Tile<pto::TileType::Right, half, 128, 128, pto::BLayout::RowMajor,
                              128, 128, pto::SLayout::ColMajor, 512>;
  using WideAcc = pto::Tile<pto::TileType::Acc, float, 16, 128, pto::BLayout::ColMajor,
                            16, 128, pto::SLayout::RowMajor, 1024>;
  PageMat wideKey;  // One 32-KiB native KV-head page in L1 [32768,65536).
  WideRight wideRight;
  WideAcc wideAcc;
  pto::TASSIGN<32768>(wideKey);
  pto::TASSIGN<0>(wideRight);
  pto::TASSIGN<0>(wideAcc);
#endif
  BlockFloat cubeValues;
  BlockHalf cubeHalf;
  Rows scores, exponential, scratch, expanded, rescale, numerator, pageValue;
  HalfRows halfScores, probabilities, result, halfDivisor;
  PaddedHalf paddedProbability;
  Column maximum, oldMaximum, pageSum, total;
  Column totalFactors;
  using Factors = pto::Tile<pto::TileType::Vec, float, 1, 16>;
  using HalfFactors = pto::Tile<pto::TileType::Vec, half, 1, 16>;
  Factors maximumFactors, oldMaximumFactors, rescaleFactors;
  HalfFactors halfFactors;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<32768>(mb);
  pto::TASSIGN<0>(left);
  pto::TASSIGN<0>(right);
  pto::TASSIGN<0>(acc);
  pto::TASSIGN<0>(cubeValues);
  pto::TASSIGN<1024>(cubeHalf);
  pto::TASSIGN<2048>(scores);
  pto::TASSIGN<6144>(halfScores);
  pto::TASSIGN<8192>(probabilities);
  pto::TASSIGN<8192>(paddedProbability);
  pto::TASSIGN<12288>(exponential);
  pto::TASSIGN<16384>(scratch);
  pto::TASSIGN<62464>(maximum);
  pto::TASSIGN<62528>(oldMaximum);
  pto::TASSIGN<20544>(pageSum);
  pto::TASSIGN<20576>(total);
  pto::TASSIGN<62592>(totalFactors);
  pto::TASSIGN<62464>(maximumFactors);
  pto::TASSIGN<62528>(oldMaximumFactors);
  pto::TASSIGN<62592>(rescaleFactors);
  pto::TASSIGN<1024>(halfFactors);
  pto::TASSIGN<20736>(expanded);
  pto::TASSIGN<24832>(rescale);
  pto::TASSIGN<28928>(numerator);
  pto::TASSIGN<33024>(pageValue);
  pto::TASSIGN<37120>(result);
  pto::TASSIGN<39168>(halfDivisor);
#ifdef PYPTO_A1_RETAIN_PROB_PADDING
  using ProbabilityPadding = pto::Tile<pto::TileType::Vec, float, 8, 64>;
  ProbabilityPadding probabilityPadding;
  pto::TASSIGN<10240>(probabilityPadding);
  pto::TEXPANDS(probabilityPadding, 0.0f);
  pipe_barrier(PIPE_ALL);
#endif
#ifdef PYPTO_A1_DIRECT_PAGES
#ifdef PYPTO_A1_WIDE_PV
  using ZeroPage = pto::Tile<pto::TileType::Vec, float, 1, 1024>;
  ZeroPage zero;
#else
  BlockFloat zero;
#endif
  pto::TASSIGN<41216>(zero);
  pto::TEXPANDS(zero, 0.0f);
  pipe_barrier(PIPE_ALL);
#endif
  pto::TEXPANDS(maximumFactors, 0.0f);
  pto::TEXPANDS(oldMaximumFactors, 0.0f);
  pipe_barrier(PIPE_ALL);
  for (unsigned task = block; task < 80; task += 8) {
    const int32_t length = Data<int32_t>(lengths)[task / 4 * 8];
    if (length < 0 || length > 4096) {
      PublishStatus(Data<int32_t>(status) + task * 16, 1);
      continue;
    }
#ifdef PYPTO_A1_DIRECT_PAGES
    bool valid = true;
    for (unsigned page = 0; page < (static_cast<unsigned>(length) + 127) / 128; ++page) {
      const int32_t physical = Data<int32_t>(table)[task / 4 * 32 + page];
      if (physical < 0 || static_cast<unsigned>(physical) >= keys->shapes[0]) valid = false;
    }
    if (!valid) {
      PublishStatus(Data<int32_t>(status) + task * 16, 2);
      continue;
    }
#endif
#ifdef PYPTO_A1_WIDE_QK
    // Keep the eight original QK K16 blocks in a separate L1 region.
    // PV reuses ma at address zero, so it must not overwrite the resident Q.
    for (unsigned k = 0; k < 128; k += 16) {
      Mat blockQuery;
      pto::TASSIGN(blockQuery, 65536 + k / 16 * 512);
      MatrixGlobal q(Data<half>(query) + task * 16 * 128 + k);
      pto::TLOAD(blockQuery, q);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (unsigned k = 0; k < 128; k += 16) {
      load_cbuf_to_ca(queryLeft.data() + k / 16 * 256,
                      queryCache.data() + k / 16 * 256,
                      0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
    }
#endif
    pto::TEXPANDS(numerator, 0.0f);
    pto::TEXPANDS(total, 0.0f);
    pipe_barrier(PIPE_ALL);
    for (unsigned page = 0; page < static_cast<unsigned>(length); page += 128) {
#ifdef PYPTO_A1_DIRECT_PAGES
      const uint64_t physical = Data<int32_t>(table)[task / 4 * 32 + page / 128];
      const uint64_t pageBase = (physical * 32 + task % 4 * 8) * 128 * 16;
#endif
#ifdef PYPTO_A1_WIDE_QK
      // Retain the full K128 operand in L0B before issuing the QK matrix operation.
      // Native page blocks already have the required QK orientation.
      copy_gm_to_cbuf(wideKey.data(), Data<half>(keys) + pageBase,
                     0, 1, 1024, 0, 0, (pad_t)0);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      for (unsigned k = 0; k < 128; k += 16) {
        load_cbuf_to_cb(wideRight.data() + k / 16 * 128 * 16,
                        wideKey.data() + k / 16 * 128 * 16,
                        0, 8, 1, 0, 0, false, addr_cal_mode_t(0));
      }
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#ifdef PYPTO_A1_PREFETCH_VALUES
      // M waits for K in L0B. Separately fence MTE2 against MTE1 before
      // overwriting the L1 K page; an M-targeted event does not fence MTE2.
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // V prefetch and QK MAD can then use independent L1/L0B storage.
      copy_gm_to_cbuf(wideKey.data(), Data<half>(values) + pageBase,
                     0, 1, 1024, 0, 0, (pad_t)0);
#endif
      mad(wideAcc.data(), queryLeft.data(), wideRight.data(), 16, 128, 128, true);
      pipe_barrier(PIPE_ALL);
#ifdef PYPTO_A1_VECTOR_COPY
      using WideScoreValues = pto::Tile<pto::TileType::Vec, float, 1, 2048>;
      using WideScoreHalves = pto::Tile<pto::TileType::Vec, half, 1, 2048>;
      WideScoreValues wideScoreValues;
      WideScoreHalves wideScoreHalves;
      // Disjoint from persistent online-softmax state and the zero-page buffer.
      pto::TASSIGN<49152>(wideScoreValues);
      pto::TASSIGN<57344>(wideScoreHalves);
      copy_matrix_cc_to_ubuf(wideScoreValues.data(), wideAcc.data(),
                             0, 1, 8, 0, 0, CRMODE_NONE);
      pipe_barrier(PIPE_ALL);
      pto::TCVT(wideScoreHalves, wideScoreValues, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(wideScoreValues, wideScoreHalves, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      for (unsigned head = 0; head < 8; ++head) {
        copy_ubuf_to_ubuf(scores.data() + head * 128, wideScoreValues.data() + head * 16,
                         0, 8, 2, 30, 0);
      }
      pipe_barrier(PIPE_ALL);
#else
      for (unsigned column = 0; column < 128; column += 16) {
        copy_matrix_cc_to_ubuf(cubeValues.data(), wideAcc.data() + column / 16 * 256,
                               0, 1, 1, 0, 0, CRMODE_NONE);
        pipe_barrier(PIPE_V);
        pto::TCVT(cubeHalf, cubeValues, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(cubeValues, cubeHalf, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (unsigned head = 0; head < 8; ++head)
          for (unsigned lane = 0; lane < 16; ++lane)
            scores.data()[head * 128 + column + lane] = cubeValues.data()[head * 16 + lane];
        pipe_barrier(PIPE_ALL);
      }
#endif
#else
      for (unsigned column = 0; column < 128; column += 16) {
        for (unsigned k = 0; k < 128; k += 16) {
          MatrixGlobal q(Data<half>(query) + task * 16 * 128 + k);
#ifdef PYPTO_A1_DIRECT_PAGES
          CacheGlobal key(Data<half>(keys) + pageBase + k / 16 * 128 * 16 + column * 16);
#else
          MatrixGlobal key(Data<half>(keys) + task * 4096 * 128 + (page + column) * 128 + k);
#endif
          pto::TLOAD(ma, q);
          pto::TLOAD(mb, key);
#ifdef PYPTO_A1_DIRECT_PAGES
          // Preserve the staging path's zero padding, including NaN cache tails.
          if (page + column + 16 > static_cast<unsigned>(length)) {
            pipe_barrier(PIPE_ALL);
            for (unsigned lane = 0; lane < 16; ++lane) {
              if (page + column + lane >= static_cast<unsigned>(length))
                copy_ubuf_to_cbuf(mb.data() + lane * 16, reinterpret_cast<__ubuf__ half*>(zero.data()),
                                  0, 1, 1, 0, 0);
            }
            pipe_barrier(PIPE_ALL);
          }
#endif
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          pto::TMOV(left, ma);
          // K is loaded as [position, dimension]. Unlike ND KxN weights,
          // this operand already has the physical ZN orientation for QK^T.
          load_cbuf_to_cb(right.data(), mb.data(), 0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (k == 0) pto::TMATMUL(acc, left, right);
          else pto::TMATMUL_ACC(acc, acc, left, right);
          pipe_barrier(PIPE_ALL);
        }
        pto::TMOV<BlockFloat, Acc, pto::AccToVecMode::SingleModeVec0>(cubeValues, acc);
        pipe_barrier(PIPE_V);
        pto::TCVT(cubeHalf, cubeValues, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        pto::TCVT(cubeValues, cubeHalf, pto::RoundMode::CAST_NONE);
#ifdef PYPTO_A1_VECTOR_COPY
        CopyHeadColumns(scores.data() + column, cubeValues.data());
#else
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (unsigned head = 0; head < 8; ++head)
          for (unsigned lane = 0; lane < 16; ++lane)
            scores.data()[head * 128 + column + lane] = cubeValues.data()[head * 16 + lane];
        pipe_barrier(PIPE_ALL);
#endif
      }
#endif
      pto::TMULS(scores, scores, 0.08837890625f);
      pipe_barrier(PIPE_V);
      pto::TCVT(halfScores, scores, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(scores, halfScores, pto::RoundMode::CAST_NONE);
#ifdef PYPTO_A1_VECTOR_COPY
      if (page + 128 > static_cast<unsigned>(length)) {
        const unsigned valid = static_cast<unsigned>(length) - page;
        const uint64_t low = valid < 64 ? UINT64_MAX << valid : 0;
        const uint64_t high = valid <= 64 ? UINT64_MAX : UINT64_MAX << (valid - 64);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        if (low) {
          set_vector_mask(0, low);
          vector_dup(scores.data(), -3.402823466e38f, 8, 1, 1, 16, 0);
          pipe_barrier(PIPE_V);
        }
        set_vector_mask(0, high);
        vector_dup(scores.data() + 64, -3.402823466e38f, 8, 1, 1, 16, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
      }
#else
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (unsigned head = 0; head < 8; ++head)
        for (unsigned lane = 0; lane < 128; ++lane)
          if (page + lane >= static_cast<unsigned>(length)) scores.data()[head * 128 + lane] = -3.402823466e38f;
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
#endif
#ifdef PYPTO_A1_VECTOR_COPY
      pto::TROWMAX(maximum, scores, scratch);
      pipe_barrier(PIPE_V);
      if (page == 0) {
        pto::TMOV(oldMaximumFactors, maximumFactors);
        pipe_barrier(PIPE_ALL);
      } else {
        set_mask_norm();
        set_vector_mask(0, 65535ULL);
        vmax(maximumFactors.data(), oldMaximumFactors.data(), maximumFactors.data(),
             1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        pipe_barrier(PIPE_V);
      }
#else
      pto::TROWMAX(maximum, scores, scratch);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      if (page == 0) {
        for (unsigned head = 0; head < 8; ++head) oldMaximum.data()[head] = maximum.data()[head];
      } else {
        for (unsigned head = 0; head < 8; ++head)
          if (oldMaximum.data()[head] > maximum.data()[head]) maximum.data()[head] = oldMaximum.data()[head];
      }
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
#endif
      // Eight head factors in aligned 16-lane storage.
      pto::TSUB(rescaleFactors, oldMaximumFactors, maximumFactors);
      pipe_barrier(PIPE_V);
      pto::TCVT(halfFactors, rescaleFactors, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(rescaleFactors, halfFactors, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TEXP(rescaleFactors, rescaleFactors);
      pipe_barrier(PIPE_V);
      pto::TROWEXPAND(rescale, totalFactors);
      pto::TROWEXPAND(expanded, maximum);
      pipe_barrier(PIPE_V);
      pto::TSUB(scores, scores, expanded);
      pipe_barrier(PIPE_V);
      pto::TCVT(halfScores, scores, pto::RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
      pto::TCVT(scores, halfScores, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      pto::TEXP(exponential, scores);
      pipe_barrier(PIPE_V);
#ifdef PYPTO_A1_VECTOR_COPY
      pto::TROWSUM(pageSum, exponential, scratch);
      pto::TCVT(probabilities, exponential, pto::RoundMode::CAST_RINT);
#ifdef PYPTO_A1_ATTENTION_LEGACY_SCALAR_TOTAL
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (unsigned head = 0; head < 8; ++head) {
        total.data()[head] = total.data()[head] * rescale.data()[head * 128] + pageSum.data()[head];
        oldMaximum.data()[head] = maximum.data()[head];
      }
#else
      // Copy the aligned state without scalar loads/stores or new arithmetic.
      // Keep separate native multiply/add rounding in the denominator update.
      pipe_barrier(PIPE_ALL);
      pto::TMOV(oldMaximumFactors, maximumFactors);
      pipe_barrier(PIPE_ALL);
      pto::TMUL(total, total, totalFactors);
      pipe_barrier(PIPE_V);
      pto::TADD(total, total, pageSum);
      pipe_barrier(PIPE_ALL);
#endif
#else
      pto::TROWSUM(pageSum, exponential, scratch);
      pto::TCVT(probabilities, exponential, pto::RoundMode::CAST_RINT);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (unsigned head = 0; head < 8; ++head) {
#ifdef PYPTO_A1_ATTENTION_LEGACY_SCALAR_TOTAL
        total.data()[head] = total.data()[head] * rescale.data()[head * 128] + pageSum.data()[head];
#endif
        oldMaximum.data()[head] = maximum.data()[head];
      }
#ifndef PYPTO_A1_ATTENTION_LEGACY_SCALAR_TOTAL
      // Native PA rounds the vector multiply before its vector add. A scalar
      // multiply-add can differ by one FP32 ULP exactly at an FP16 midpoint.
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      pto::TMUL(total, total, totalFactors);
      pipe_barrier(PIPE_V);
      pto::TADD(total, total, pageSum);
      pipe_barrier(PIPE_ALL);
#endif
#endif
#ifndef PYPTO_A1_RETAIN_PROB_PADDING
      auto* probabilityBits = reinterpret_cast<__ubuf__ uint16_t*>(paddedProbability.data());
      for (unsigned element = 8 * 128; element < 16 * 128; ++element) probabilityBits[element] = 0;
      pipe_barrier(PIPE_ALL);
#endif
#ifdef PYPTO_A1_WIDE_PV
      // Reuse each probability tile across the complete head dimension.
      // Reuse the L1 page region after QK finishes. Each V dimension block
      // has eight K16 fragments; the load2d source stride selects the same
      // fragments as the original eight sparse GM gathers.
#ifndef PYPTO_A1_PREFETCH_VALUES
      copy_gm_to_cbuf(wideKey.data(), Data<half>(values) + pageBase,
                     0, 1, 1024, 0, 0, (pad_t)0);
#endif
      pipe_barrier(PIPE_ALL);
      if (page + 128 > static_cast<unsigned>(length)) {
        const unsigned valid = static_cast<unsigned>(length) - page;
        for (unsigned sub = 0; sub < 8; ++sub) {
          copy_ubuf_to_cbuf(wideKey.data() + sub * 128 * 16 + valid * 16,
                            reinterpret_cast<__ubuf__ half*>(zero.data()),
                            0, 1, 128 - valid, 0, 0);
        }
        pipe_barrier(PIPE_ALL);
      }
      using ProbabilityMat = pto::Tile<pto::TileType::Mat, half, 16, 128,
          pto::BLayout::RowMajor, 16, 128, pto::SLayout::RowMajor, 512>;
      using ProbabilityLeft = pto::Tile<pto::TileType::Left, half, 16, 128,
          pto::BLayout::ColMajor, 16, 128, pto::SLayout::RowMajor, 512>;
      ProbabilityMat probabilityCache;
      ProbabilityLeft probabilityLeft;
      pto::TASSIGN<0>(probabilityCache);
      pto::TASSIGN<0>(probabilityLeft);
      for (unsigned k = 0; k < 128; k += 16) {
        copy_ubuf_to_cbuf(probabilityCache.data() + k / 16 * 256,
                          paddedProbability.data() + k, 0, 16, 1, 7, 0);
      }
      pipe_barrier(PIPE_ALL);
      load_cbuf_to_ca(probabilityLeft.data(), probabilityCache.data(),
                      0, 8, 1, 0, 0, false, addr_cal_mode_t(0));
      for (unsigned k = 0; k < 128; k += 16) {
        load_cbuf_to_cb(wideRight.data() + k / 16 * 128 * 16,
                        wideKey.data() + k * 16,
                        0, 8, 8, 0, 0, true, addr_cal_mode_t(0));
      }
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      mad(wideAcc.data(), probabilityLeft.data(), wideRight.data(), 16, 128, 128, true);
      pipe_barrier(PIPE_ALL);
#if defined(PYPTO_A1_VECTOR_COPY) && defined(PYPTO_A1_WIDE_QK)
      // QK score packing has finished; reuse its private NZ scratch for FP32 PV.
      copy_matrix_cc_to_ubuf(wideScoreValues.data(), wideAcc.data(),
                             0, 1, 8, 0, 0, CRMODE_NONE);
      pipe_barrier(PIPE_ALL);
      for (unsigned head = 0; head < 8; ++head) {
        copy_ubuf_to_ubuf(pageValue.data() + head * 128, wideScoreValues.data() + head * 16,
                         0, 8, 2, 30, 0);
      }
      pipe_barrier(PIPE_ALL);
#else
      for (unsigned column = 0; column < 128; column += 16) {
        copy_matrix_cc_to_ubuf(cubeValues.data(), wideAcc.data() + column / 16 * 256,
                              0, 1, 1, 0, 0, CRMODE_NONE);
#ifdef PYPTO_A1_VECTOR_COPY
        CopyHeadColumns(pageValue.data() + column, cubeValues.data());
#else
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (unsigned head = 0; head < 8; ++head)
          for (unsigned lane = 0; lane < 16; ++lane)
            pageValue.data()[head * 128 + column + lane] = cubeValues.data()[head * 16 + lane];
        pipe_barrier(PIPE_ALL);
#endif
      }
#endif
#else
      for (unsigned column = 0; column < 128; column += 16) {
        for (unsigned k = 0; k < 128; k += 16) {
          copy_ubuf_to_cbuf(ma.data(), paddedProbability.data() + k, 0, 16, 1, 7, 0);
#ifdef PYPTO_A1_DIRECT_PAGES
          CacheGlobal value(Data<half>(values) + pageBase + column / 16 * 128 * 16 + k * 16);
#else
          MatrixGlobal value(Data<half>(values) + task * 4096 * 128 + (page + k) * 128 + column);
#endif
          pto::TLOAD(mb, value);
          pipe_barrier(PIPE_ALL);
#ifdef PYPTO_A1_DIRECT_PAGES
          if (page + k + 16 > static_cast<unsigned>(length)) {
            for (unsigned lane = 0; lane < 16; ++lane) {
              if (page + k + lane >= static_cast<unsigned>(length))
                copy_ubuf_to_cbuf(mb.data() + lane * 16, reinterpret_cast<__ubuf__ half*>(zero.data()),
                                  0, 1, 1, 0, 0);
            }
            pipe_barrier(PIPE_ALL);
          }
#endif
          pto::TMOV(left, ma);
          pto::TMOV(right, mb);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (k == 0) pto::TMATMUL(acc, left, right);
          else pto::TMATMUL_ACC(acc, acc, left, right);
          pipe_barrier(PIPE_ALL);
        }
        pto::TMOV<BlockFloat, Acc, pto::AccToVecMode::SingleModeVec0>(cubeValues, acc);
#ifdef PYPTO_A1_VECTOR_COPY
        CopyHeadColumns(pageValue.data() + column, cubeValues.data());
#else
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (unsigned head = 0; head < 8; ++head)
          for (unsigned lane = 0; lane < 16; ++lane)
            pageValue.data()[head * 128 + column + lane] = cubeValues.data()[head * 16 + lane];
        pipe_barrier(PIPE_ALL);
#endif
      }
#endif
      pto::TMUL(numerator, numerator, rescale);
      pipe_barrier(PIPE_V);
      pto::TADD(numerator, numerator, pageValue);
      pipe_barrier(PIPE_ALL);
    }
    if (length == 0) {
      for (unsigned head = 0; head < 8; ++head) total.data()[head] = 1.0f;
      pipe_barrier(PIPE_ALL);
    }
    pto::TROWEXPAND(expanded, total);
    pipe_barrier(PIPE_V);
    pto::TCVT(halfDivisor, expanded, pto::RoundMode::CAST_RINT);
    pto::TCVT(result, numerator, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(UINT64_MAX, UINT64_MAX);
    vdiv(result.data(), result.data(), halfDivisor.data(), 8, 1, 1, 1, 8, 8, 8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    OutputGlobal destination(Data<half>(output) + task * 8 * 128);
    pto::TSTORE(destination, result);
    pipe_barrier(PIPE_ALL);
#ifdef PYPTO_A1_ATTENTION_DIAGNOSTIC
#ifdef PYPTO_A1_DIRECT_PAGES
    auto* diagnosticNumerator = reinterpret_cast<__gm__ Tensor*>(args[7]);
#else
    auto* diagnosticNumerator = reinterpret_cast<__gm__ Tensor*>(args[6]);
#endif
    using DiagnosticGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>,
                                              pto::Stride<1, 1, 1, 128, 1>>;
    DiagnosticGlobal diagnosticOutput(Data<float>(diagnosticNumerator) + task * 8 * 128);
    pto::TSTORE(diagnosticOutput, numerator);
    pipe_barrier(PIPE_ALL);
    auto* diagnostic = Data<float>(status) + task * 16;
    for (unsigned head = 0; head < 8; ++head) diagnostic[1 + head] = total.data()[head];
    for (unsigned head = 0; head < 7; ++head) diagnostic[9 + head] = numerator.data()[head * 128];
#endif
    PublishStatus(Data<int32_t>(status) + task * 16, 0);
  }
}
