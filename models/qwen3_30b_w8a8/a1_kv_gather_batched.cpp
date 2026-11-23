/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_cache_common.h"

constexpr unsigned BATCH = 20;
constexpr unsigned PAGE = 128;
constexpr unsigned CHUNK = 16;
constexpr unsigned GROUPS = WIDTH / CHUNK;
constexpr unsigned TOKEN_TILE = 8;
using CopyTile = pto::Tile<pto::TileType::Vec, bfloat16_t, TOKEN_TILE, CHUNK>;
using InputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, TOKEN_TILE, CHUNK>,
                                     pto::Stride<1, 1, 1, CHUNK, 1>>;
using OutputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, TOKEN_TILE, CHUNK>,
                                      pto::Stride<1, 1, 1, HEAD_DIM, 1>>;
using ZeroTile = pto::Tile<pto::TileType::Vec, float, TOKEN_TILE, CHUNK / 2>;
using ZeroRow = pto::Tile<pto::TileType::Vec, float, 1, CHUNK / 2>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  auto* keyCache = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* valueCache = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* tableTensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* lengths = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* keys = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* values = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
  CopyTile keyTile, valueTile;
  ZeroTile zeroKey, zeroValue;
  pto::TASSIGN<0>(keyTile);
  pto::TASSIGN<256>(valueTile);
  pto::TASSIGN<0>(zeroKey);
  pto::TASSIGN<256>(zeroValue);
  for (unsigned task = get_block_idx(); task < BATCH * HEADS; task += get_block_num()) {
    const unsigned row = task / HEADS, head = task % HEADS;
    auto* status = Data<int32_t>(statusTensor) + task * 16;
    if (keyCache->ndims != 4 || valueCache->ndims != 4 ||
        keyCache->shapes[0] != valueCache->shapes[0] || keyCache->shapes[1] != GROUPS ||
        valueCache->shapes[1] != GROUPS || keyCache->shapes[2] != PAGE ||
        valueCache->shapes[2] != PAGE || keyCache->shapes[3] != CHUNK || valueCache->shapes[3] != CHUNK ||
        keys->ndims != 3 || values->ndims != 3 || keys->shapes[0] != BATCH * HEADS ||
        values->shapes[0] != BATCH * HEADS || keys->shapes[1] != values->shapes[1] ||
        keys->shapes[1] == 0 || keys->shapes[1] % PAGE || keys->shapes[1] > 4096 ||
        keys->shapes[2] != HEAD_DIM || values->shapes[2] != HEAD_DIM ||
        tableTensor->ndims != 2 || tableTensor->shapes[0] != BATCH ||
        tableTensor->shapes[1] < keys->shapes[1] / PAGE || lengths->ndims != 2 ||
        lengths->shapes[0] != BATCH || lengths->shapes[1] != 8) {
      PublishStatus(status, 3);
      continue;
    }
    const unsigned capacity = keys->shapes[1];
    const int32_t length = Data<int32_t>(lengths)[row * 8];
    if (length < 0 || static_cast<unsigned>(length) > capacity) {
      PublishStatus(status, 1);
      continue;
    }
    auto* table = Data<int32_t>(tableTensor) + row * tableTensor->shapes[1];
    bool valid = true;
    for (unsigned page = 0; page < (static_cast<unsigned>(length) + PAGE - 1) / PAGE; ++page) {
      if (table[page] < 0 || static_cast<unsigned>(table[page]) >= keyCache->shapes[0]) valid = false;
    }
    if (!valid) {
      PublishStatus(status, 2);
      continue;
    }
    for (unsigned token = 0; token < capacity; token += TOKEN_TILE) {
      for (unsigned chunk = 0; chunk < HEAD_DIM / CHUNK; ++chunk) {
        if (token < static_cast<unsigned>(length)) {
          const uint64_t offset = ((static_cast<uint64_t>(table[token / PAGE]) * GROUPS +
                                    head * (HEAD_DIM / CHUNK) + chunk) * PAGE + token % PAGE) * CHUNK;
          InputGlobal srcKey(Data<bfloat16_t>(keyCache) + offset);
          InputGlobal srcValue(Data<bfloat16_t>(valueCache) + offset);
          pto::TLOAD(keyTile, srcKey);
          pto::TLOAD(valueTile, srcValue);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          for (unsigned tail = 0; tail < TOKEN_TILE; ++tail) {
            if (token + tail >= static_cast<unsigned>(length)) {
              ZeroRow tailKey, tailValue;
              pto::TASSIGN(tailKey, reinterpret_cast<uint64_t>(keyTile.data() + tail * CHUNK));
              pto::TASSIGN(tailValue, reinterpret_cast<uint64_t>(valueTile.data() + tail * CHUNK));
              pto::TEXPANDS(tailKey, 0.0f);
              pto::TEXPANDS(tailValue, 0.0f);
            }
          }
        } else {
          pto::TEXPANDS(zeroKey, 0.0f);
          pto::TEXPANDS(zeroValue, 0.0f);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        const uint64_t offset = (static_cast<uint64_t>(task) * capacity + token) * HEAD_DIM + chunk * CHUNK;
        OutputGlobal dstKey(Data<bfloat16_t>(keys) + offset);
        OutputGlobal dstValue(Data<bfloat16_t>(values) + offset);
        pto::TSTORE(dstKey, keyTile);
        pto::TSTORE(dstValue, valueTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
    }
    PublishStatus(status, 0);
  }
}
