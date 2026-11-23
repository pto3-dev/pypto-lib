/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* q = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* k = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* packedQ = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* packedK = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (q->ndims != 2 || q->shapes[0] != 640 || q->shapes[1] != 128 ||
      k->ndims != 2 || k->shapes[0] != 160 || k->shapes[1] != 128 ||
      packedQ->ndims != 3 || packedQ->shapes[0] != 80 || packedQ->shapes[1] != 16 || packedQ->shapes[2] != 128 ||
      packedK->ndims != 2 || packedK->shapes[0] != 20 || packedK->shapes[1] != 512 ||
      status->ndims != 2 || status->shapes[0] != 80 || status->shapes[1] != 16) {
    trap();
    return;
  }
  auto* values = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* keyCache = reinterpret_cast<__gm__ Tensor*>(args[6]);
  auto* valueCache = reinterpret_cast<__gm__ Tensor*>(args[7]);
  auto* slots = reinterpret_cast<__gm__ Tensor*>(args[8]);
  auto* appendStatus = reinterpret_cast<__gm__ Tensor*>(args[9]);
  if (values->ndims != 2 || values->shapes[0] != 20 || values->shapes[1] != 512 ||
      keyCache->ndims != 4 || valueCache->ndims != 4 ||
      keyCache->shapes[0] != valueCache->shapes[0] || keyCache->shapes[1] != 32 ||
      valueCache->shapes[1] != 32 || keyCache->shapes[2] != 128 ||
      valueCache->shapes[2] != 128 || keyCache->shapes[3] != 16 || valueCache->shapes[3] != 16 ||
      slots->ndims != 2 || slots->shapes[0] != 20 || slots->shapes[1] != 8 ||
      appendStatus->ndims != 2 || appendStatus->shapes[0] != 20 || appendStatus->shapes[1] != 16) {
    trap();
    return;
  }
  using CacheTile = pto::Tile<pto::TileType::Vec, half, 8, 16>;
  using CacheGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 16>,
                                      pto::Stride<1, 1, 1, 2048, 1>>;
  CacheTile cachedKey, cachedValue;
  pto::TASSIGN<4096>(cachedKey);
  pto::TASSIGN<4352>(cachedValue);
  using QTile = pto::Tile<pto::TileType::Vec, half, 8, 128>;
  using KTile = pto::Tile<pto::TileType::Vec, half, 1, 128>;
  using QGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  using KGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;
  QTile active, zero;
  KTile key;
  pto::TASSIGN<0>(active);
  pto::TASSIGN<2048>(zero);
  pto::TASSIGN<4096>(key);
  for (unsigned element = 0; element < 1024; ++element)
    reinterpret_cast<__ubuf__ uint16_t*>(zero.data())[element] = 0;
  pipe_barrier(PIPE_ALL);
  // Forty two-head tasks balance exactly over eight cores. Validate each slot
  // once per pair instead of repeating the same twenty-slot scan for every head.
  for (unsigned group = block; group < 40; group += 8) {
    const unsigned row = group / 2;
    const unsigned firstHead = (group % 2) * 2;
    const int32_t slot = Data<int32_t>(slots)[row * 8];
    int32_t code = 0;
    if (slot >= 0) {
      if (static_cast<uint64_t>(slot) >= static_cast<uint64_t>(keyCache->shapes[0]) * 128) {
        code = 1;
      } else {
        for (unsigned other = 0; other < 20; ++other)
          if (other != row && Data<int32_t>(slots)[other * 8] == slot) code = 3;
      }
    }
    for (unsigned head = firstHead; head < firstHead + 2; ++head) {
      const unsigned task = row * 4 + head;
      QGlobal source(Data<half>(q) + task * 1024);
      KGlobal keySource(Data<half>(k) + (task / 4 * 8 + task % 4) * 128);
      pto::TLOAD(active, source);
      pto::TLOAD(key, keySource);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      QGlobal activeOutput(Data<half>(packedQ) + task * 2048);
      QGlobal padding(Data<half>(packedQ) + task * 2048 + 1024);
      KGlobal keyOutput(Data<half>(packedK) + task * 128);
      pto::TSTORE(activeOutput, active);
      pto::TSTORE(padding, zero);
      pto::TSTORE(keyOutput, key);
      pipe_barrier(PIPE_ALL);
      if (slot >= 0 && code == 0) {
        KGlobal valueSource(Data<half>(values) + task * 128);
        KTile value;
        pto::TASSIGN<4352>(value);
        pto::TLOAD(value, valueSource);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        const uint64_t offset = (static_cast<uint64_t>(slot / 128) * 32 * 128 +
                                 head * 8 * 128 + slot % 128) * 16;
        CacheGlobal keyOutput(Data<half>(keyCache) + offset);
        CacheGlobal valueOutput(Data<half>(valueCache) + offset);
        pto::TSTORE(keyOutput, cachedKey);
        pto::TSTORE(valueOutput, cachedValue);
        pipe_barrier(PIPE_ALL);
      }
      PublishStatus(Data<int32_t>(status) + task * 16, 0);
    }
    if (firstHead == 0) PublishStatus(Data<int32_t>(appendStatus) + row * 16, code);
  }
}
