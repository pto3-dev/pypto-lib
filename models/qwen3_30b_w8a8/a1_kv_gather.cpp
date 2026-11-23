/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_cache_common.h"

constexpr unsigned PAGE = 128;
constexpr unsigned TOKEN_TILE = 8;
using GatherTile = pto::Tile<pto::TileType::Vec, bfloat16_t, TOKEN_TILE, HEAD_DIM>;
using InputGlobal =
    pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, TOKEN_TILE, HEAD_DIM>, pto::Stride<1, 1, 1, WIDTH, 1>>;
using OutputGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, TOKEN_TILE, HEAD_DIM>,
                                       pto::Stride<1, 1, 1, HEAD_DIM, 1>>;
using ZeroTile = pto::Tile<pto::TileType::Vec, float, TOKEN_TILE, HEAD_DIM / 2>;
using ZeroRow = pto::Tile<pto::TileType::Vec, float, 1, HEAD_DIM / 2>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned head = get_block_idx();
  if (head >= HEADS) return;
  auto* keyCache = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* valueCache = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* tableTensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* lengthTensor = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* keys = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* values = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
  auto* status = Data<int32_t>(statusTensor) + head * 16;
  if (keyCache->ndims != 2 || valueCache->ndims != 2 || keys->ndims != 2 || values->ndims != 2 ||
      keyCache->shapes[0] != valueCache->shapes[0] || keyCache->shapes[1] != WIDTH ||
      valueCache->shapes[1] != WIDTH || keyCache->shapes[0] % PAGE || keys->shapes[0] != values->shapes[0] ||
      keys->shapes[0] % (HEADS * PAGE) || keys->shapes[1] != HEAD_DIM || values->shapes[1] != HEAD_DIM) {
    PublishStatus(status, 3);
    return;
  }
  const unsigned capacity = keys->shapes[0] / HEADS;
  const int32_t length = Data<int32_t>(lengthTensor)[0];
  if (length < 0 || static_cast<uint32_t>(length) > capacity) {
    PublishStatus(status, 1);
    return;
  }
  auto* table = Data<int32_t>(tableTensor);
  const unsigned pages = (static_cast<unsigned>(length) + PAGE - 1) / PAGE;
  for (unsigned page = 0; page < pages; ++page) {
    if (table[page] < 0 || static_cast<uint32_t>(table[page]) >= keyCache->shapes[0] / PAGE) {
      PublishStatus(status, 2);
      return;
    }
  }
  GatherTile keyTile, valueTile;
  ZeroTile zeroKey, zeroValue;
  pto::TASSIGN<0>(keyTile);
  pto::TASSIGN<2048>(valueTile);
  pto::TASSIGN<0>(zeroKey);
  pto::TASSIGN<2048>(zeroValue);
  for (unsigned token = 0; token < capacity; token += TOKEN_TILE) {
    if (token < static_cast<unsigned>(length)) {
      const uint64_t physical = static_cast<uint64_t>(table[token / PAGE]) * PAGE + token % PAGE;
      const uint64_t offset = physical * WIDTH + head * HEAD_DIM;
      InputGlobal sourceKey(Data<bfloat16_t>(keyCache) + offset);
      InputGlobal sourceValue(Data<bfloat16_t>(valueCache) + offset);
      pto::TLOAD(keyTile, sourceKey);
      pto::TLOAD(valueTile, sourceValue);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      for (unsigned row = 0; row < TOKEN_TILE; ++row) {
        if (token + row >= static_cast<unsigned>(length)) {
          ZeroRow tailKey, tailValue;
          pto::TASSIGN(tailKey, reinterpret_cast<uint64_t>(keyTile.data() + row * HEAD_DIM));
          pto::TASSIGN(tailValue, reinterpret_cast<uint64_t>(valueTile.data() + row * HEAD_DIM));
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
    const uint64_t offset = (static_cast<uint64_t>(head) * capacity + token) * HEAD_DIM;
    OutputGlobal targetKey(Data<bfloat16_t>(keys) + offset);
    OutputGlobal targetValue(Data<bfloat16_t>(values) + offset);
    pto::TSTORE(targetKey, keyTile);
    pto::TSTORE(targetValue, valueTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  PublishStatus(status, 0);
}
