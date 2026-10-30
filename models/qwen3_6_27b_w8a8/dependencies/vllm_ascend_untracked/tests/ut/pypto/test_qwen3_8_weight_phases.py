# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
"""Layout conversion must be lossless and distinguish global from tile-local NZ."""

import unittest
import torch
from vllm_ascend.pypto.qwen3_8_shared_weight_phases import pack_projection, unpack_projection


class WeightPhaseContracts(unittest.TestCase):
    def test_both_layouts_restore_all_bf16_weight_bits(self):
        torch.manual_seed(121012)
        value = torch.randn(512, 768, dtype=torch.bfloat16)
        for nz in (False, True):
            packed = pack_projection(value, nz)
            restored = unpack_projection(packed, 512, 768, nz)
            self.assertTrue(torch.equal(value, restored))

    def test_aligned_row_chunks_preserve_complete_layout_and_restore(self):
        value = torch.randn(768, 512, dtype=torch.bfloat16)
        for nz in (False, True):
            whole = pack_projection(value, nz)
            chunked = torch.cat([pack_projection(chunk, nz) for chunk in value.split(256, dim=0)])
            self.assertTrue(torch.equal(whole, chunked))
            restored = torch.cat(
                [unpack_projection(chunk, 256, 512, nz) for chunk in chunked.split(256 * 512)]
            )
            self.assertTrue(torch.equal(value, restored))

    def test_tile_local_nz_is_not_a_global_nz_alias(self):
        value = torch.arange(512 * 512, dtype=torch.int32).reshape(512, 512)
        actual = pack_projection(value, True)
        global_nz = value.reshape(32, 16, 32, 16).permute(2, 0, 1, 3).contiguous().reshape(-1)
        self.assertFalse(torch.equal(actual, global_nz))
        self.assertTrue(torch.equal(value, unpack_projection(actual, 512, 512, True)))


if __name__ == "__main__":
    unittest.main()
