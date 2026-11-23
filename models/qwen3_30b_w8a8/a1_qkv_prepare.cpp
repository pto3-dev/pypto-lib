/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Heads = pto::Tile<pto::TileType::Vec, bfloat16_t, 4, 128>;
using Rope = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 128>;
using HeadGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using RopeGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  if (get_block_idx() != 0) return;
  auto* qkv = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* position = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ropeTable = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* keyInput = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* ropeRow = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 1 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor);
  if (qkv->ndims != 2 || qkv->shapes[0] != 16 || qkv->shapes[1] != 5120 || position->ndims != 2 ||
      position->shapes[0] != 1 || position->shapes[1] != 8 || ropeTable->ndims != 2 ||
      ropeTable->shapes[0] != 512 || ropeTable->shapes[1] != 128 || keyInput->ndims != 2 ||
      keyInput->shapes[0] != 8 || keyInput->shapes[1] != 128 || ropeRow->ndims != 2 ||
      ropeRow->shapes[0] != 1 || ropeRow->shapes[1] != 128) {
    PublishStatus(status, 2);
    return;
  }
  const int32_t pos = Data<int32_t>(position)[0];
  if (pos < 0 || pos >= 512) {
    PublishStatus(status, 1);
    return;
  }
  Heads key, zeros;
  Rope rope;
  pto::TASSIGN<0>(key);
  pto::TASSIGN<1024>(zeros);
  pto::TASSIGN<2048>(rope);
  auto* zeroBits = reinterpret_cast<__ubuf__ uint16_t*>(zeros.data());
  for (unsigned lane = 0; lane < 512; ++lane) zeroBits[lane] = 0;
  HeadGlobal keySource(Data<bfloat16_t>(qkv) + 4096);
  RopeGlobal ropeSource(Data<bfloat16_t>(ropeTable) + pos * 128);
  pto::TLOAD(key, keySource);
  pto::TLOAD(rope, ropeSource);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  HeadGlobal activeKey(Data<bfloat16_t>(keyInput));
  HeadGlobal paddedKey(Data<bfloat16_t>(keyInput) + 512);
  RopeGlobal selected(Data<bfloat16_t>(ropeRow));
  pto::TSTORE(activeKey, key);
  pto::TSTORE(paddedKey, zeros);
  pto::TSTORE(selected, rope);
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
