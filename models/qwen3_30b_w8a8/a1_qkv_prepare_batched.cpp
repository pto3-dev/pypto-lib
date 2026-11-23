/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Heads = pto::Tile<pto::TileType::Vec, half, 8, 128>;
using FourHeads = pto::Tile<pto::TileType::Vec, half, 4, 128>;
using Rope = pto::Tile<pto::TileType::Vec, half, 1, 128>;
using HeadGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using FourGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 4, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using RopeGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1, 1, 1, 128, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* qkv = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* positions = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* table = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* queries = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* keys = reinterpret_cast<__gm__ Tensor*>(args[4]);
  auto* values = reinterpret_cast<__gm__ Tensor*>(args[5]);
  auto* ropeRows = reinterpret_cast<__gm__ Tensor*>(args[6]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[7]);
  if (qkv->ndims != 2 || qkv->shapes[0] != 20 || qkv->shapes[1] != 5120 ||
      positions->ndims != 2 || positions->shapes[0] != 20 || positions->shapes[1] != 8 ||
      table->ndims != 2 || table->shapes[1] != 128 || table->shapes[0] < 2448 ||
      queries->ndims != 2 || queries->shapes[0] != 640 || queries->shapes[1] != 128 ||
      keys->ndims != 2 || keys->shapes[0] != 160 || keys->shapes[1] != 128 ||
      values->ndims != 2 || values->shapes[0] != 20 || values->shapes[1] != 512 ||
      ropeRows->ndims != 2 || ropeRows->shapes[0] != 20 || ropeRows->shapes[1] != 128 ||
      status->ndims != 2 || status->shapes[0] != 20 || status->shapes[1] != 16) {
    trap();
    return;
  }
  Heads query;
  FourHeads key, value, zero;
  Rope rope;
  pto::TASSIGN<0>(query);
  pto::TASSIGN<2048>(key);
  pto::TASSIGN<3072>(value);
  pto::TASSIGN<4096>(zero);
  pto::TASSIGN<5120>(rope);
  for (unsigned element = 0; element < 512; ++element)
    reinterpret_cast<__ubuf__ uint16_t*>(zero.data())[element] = 0;
  pipe_barrier(PIPE_ALL);
  for (unsigned row = block; row < 20; row += 8) {
    const int32_t position = Data<int32_t>(positions)[row * 8];
    if (position < 0 || static_cast<uint32_t>(position) >= table->shapes[0]) {
      PublishStatus(Data<int32_t>(status) + row * 16, 1);
      continue;
    }
    for (unsigned group = 0; group < 4; ++group) {
      HeadGlobal source(Data<half>(qkv) + row * 5120 + group * 1024);
      HeadGlobal destination(Data<half>(queries) + row * 4096 + group * 1024);
      pto::TLOAD(query, source);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      pto::TSTORE(destination, query);
      pipe_barrier(PIPE_ALL);
    }
    FourGlobal keySource(Data<half>(qkv) + row * 5120 + 4096);
    FourGlobal valueSource(Data<half>(qkv) + row * 5120 + 4608);
    RopeGlobal ropeSource(Data<half>(table) + position * 128);
    pto::TLOAD(key, keySource);
    pto::TLOAD(value, valueSource);
    pto::TLOAD(rope, ropeSource);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    FourGlobal keyOut(Data<half>(keys) + row * 1024);
    FourGlobal keyPad(Data<half>(keys) + row * 1024 + 512);
    FourGlobal valueOut(Data<half>(values) + row * 512);
    RopeGlobal ropeOut(Data<half>(ropeRows) + row * 128);
    pto::TSTORE(keyOut, key);
    pto::TSTORE(keyPad, zero);
    pto::TSTORE(valueOut, value);
    pto::TSTORE(ropeOut, rope);
    pipe_barrier(PIPE_ALL);
    PublishStatus(Data<int32_t>(status) + row * 16, 0);
  }
}
