/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_cache_common.h"
using CacheTile = pto::Tile<pto::TileType::Vec, bfloat16_t, HEADS, HEAD_DIM>;
using CacheGlobal =
    pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, HEADS, HEAD_DIM>, pto::Stride<1, 1, 1, HEAD_DIM, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* key = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* value = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* keyCache = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* valueCache = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* slotTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* status = Data<int32_t>(statusTensor);
  if (key->ndims != 2 || value->ndims != 2 || keyCache->ndims != 2 || valueCache->ndims != 2 ||
      key->shapes[0] != HEADS || value->shapes[0] != HEADS || key->shapes[1] != HEAD_DIM ||
      value->shapes[1] != HEAD_DIM || keyCache->shapes[1] != WIDTH || valueCache->shapes[1] != WIDTH ||
      keyCache->shapes[0] != valueCache->shapes[0]) {
    PublishStatus(status, 2);
    return;
  }
  const int32_t slot = Data<int32_t>(slotTensor)[0];
  if (slot < 0) {
    PublishStatus(status, 0);
    return;
  }
  if (static_cast<uint32_t>(slot) >= keyCache->shapes[0]) {
    PublishStatus(status, 1);
    return;
  }
  const uint64_t offset = static_cast<uint64_t>(slot) * WIDTH;
  CacheGlobal srcKey(Data<bfloat16_t>(key)), srcValue(Data<bfloat16_t>(value));
  CacheGlobal dstKey(Data<bfloat16_t>(keyCache) + offset), dstValue(Data<bfloat16_t>(valueCache) + offset);
  CacheTile keyTile, valueTile;
  pto::TASSIGN<0>(keyTile);
  pto::TASSIGN<1024>(valueTile);
  pto::TLOAD(keyTile, srcKey);
  pto::TLOAD(valueTile, srcValue);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  pto::TSTORE(dstKey, keyTile);
  pto::TSTORE(dstValue, valueTile);
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
