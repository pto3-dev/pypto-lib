/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_cache_common.h"

constexpr unsigned BATCH = 20;
constexpr unsigned PAGE = 128;
constexpr unsigned CHUNK = 16;
constexpr unsigned GROUPS = WIDTH / CHUNK;
using CopyTile = pto::Tile<pto::TileType::Vec, bfloat16_t, GROUPS, CHUNK>;
using InputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, GROUPS, CHUNK>,
                                     pto::Stride<1, 1, 1, CHUNK, 1>>;
using OutputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, GROUPS, CHUNK>,
                                      pto::Stride<1, 1, 1, PAGE * CHUNK, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* key = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* value = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* keyCache = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* valueCache = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* slots = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
  CopyTile keyTile, valueTile;
  pto::TASSIGN<0>(keyTile);
  pto::TASSIGN<1024>(valueTile);
  for (unsigned row = get_block_idx(); row < BATCH; row += get_block_num()) {
    auto* status = Data<int32_t>(statusTensor) + row * 16;
    if (key->ndims != 2 || value->ndims != 2 || key->shapes[0] != BATCH ||
        value->shapes[0] != BATCH || key->shapes[1] != WIDTH || value->shapes[1] != WIDTH ||
        keyCache->ndims != 4 || valueCache->ndims != 4 ||
        keyCache->shapes[0] != valueCache->shapes[0] || keyCache->shapes[1] != GROUPS ||
        valueCache->shapes[1] != GROUPS || keyCache->shapes[2] != PAGE ||
        valueCache->shapes[2] != PAGE || keyCache->shapes[3] != CHUNK ||
        valueCache->shapes[3] != CHUNK || slots->ndims != 2 ||
        slots->shapes[0] != BATCH || slots->shapes[1] != 8) {
      PublishStatus(status, 2);
      continue;
    }
    const int32_t slot = Data<int32_t>(slots)[row * 8];
    if (slot < 0) {
      PublishStatus(status, 0);
      continue;
    }
    if (static_cast<uint64_t>(slot) >= static_cast<uint64_t>(keyCache->shapes[0]) * PAGE) {
      PublishStatus(status, 1);
      continue;
    }
    bool duplicate = false;
    for (unsigned other = 0; other < BATCH; ++other) {
      if (other != row && Data<int32_t>(slots)[other * 8] == slot) duplicate = true;
    }
    if (duplicate) {
      PublishStatus(status, 3);
      continue;
    }
    const uint64_t offset = (static_cast<uint64_t>(slot / PAGE) * GROUPS * PAGE + slot % PAGE) * CHUNK;
    InputGlobal srcKey(Data<bfloat16_t>(key) + row * WIDTH);
    InputGlobal srcValue(Data<bfloat16_t>(value) + row * WIDTH);
    OutputGlobal dstKey(Data<bfloat16_t>(keyCache) + offset);
    OutputGlobal dstValue(Data<bfloat16_t>(valueCache) + offset);
    pto::TLOAD(keyTile, srcKey);
    pto::TLOAD(valueTile, srcValue);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(dstKey, keyTile);
    pto::TSTORE(dstValue, valueTile);
    pipe_barrier(PIPE_ALL);
    PublishStatus(status, 0);
  }
}
