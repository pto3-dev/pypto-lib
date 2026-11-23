/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_tensor.h"

using Tokens = pto::Tile<pto::TileType::Vec, int32_t, 1, 8>;
using Row = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, 256>;
using TokenGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<1, 1, 1, 8, 1>>;
using RowGlobal = pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<1, 1, 1, 256, 1>>;

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  const unsigned partition = get_block_idx();
  if (partition >= 8) return;
  auto* tokens = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != 8 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + partition * 16;
  if (tokens->ndims != 2 || tokens->shapes[0] != 1 || tokens->shapes[1] != 8 || weight->ndims != 2 ||
      weight->shapes[1] != 2048 || weight->shapes[0] <= 0 || output->ndims != 2 || output->shapes[0] != 1 ||
      output->shapes[1] != 2048) {
    PublishStatus(status, 1);
    return;
  }
  Tokens ids;
  Row row;
  pto::TASSIGN<0>(ids);
  pto::TASSIGN<32>(row);
  TokenGlobal source(Data<int32_t>(tokens));
  pto::TLOAD(ids, source);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  const int32_t token = ids.data()[0];
  if (token < 0 || token >= weight->shapes[0]) {
    PublishStatus(status, 2);
    return;
  }
  RowGlobal embedding(Data<bfloat16_t>(weight) + static_cast<int64_t>(token) * 2048 + partition * 256);
  pto::TLOAD(row, embedding);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  RowGlobal result(Data<bfloat16_t>(output) + partition * 256);
  pto::TSTORE(result, row);
  pipe_barrier(PIPE_ALL);
  PublishStatus(status, 0);
}
