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
  for (unsigned task = block; task < 80; task += 8) {
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
    PublishStatus(Data<int32_t>(status) + task * 16, 0);
  }
}
