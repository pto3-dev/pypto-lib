# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model dimensions are independent of hardware resource quotas.

B3/B4 use the same Ascend910B ISA target. Core counts are queried by the
generated launch wrappers; capacities come from the active compiler backend.
No measured B4 core count is used as a platform limit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionConfig:
    tp: int = 4
    request_batch: int = 4
    mtp_drafts: int = 5
    hidden: int = 5120
    intermediate: int = 17408
    vocab: int = 248320
    q_heads: int = 24
    kv_heads: int = 4
    gdn_k_heads: int = 16
    gdn_v_heads: int = 48

    def __post_init__(self):
        for name in ("intermediate", "vocab", "q_heads", "kv_heads", "gdn_k_heads", "gdn_v_heads"):
            if getattr(self, name) % self.tp:
                raise ValueError(f"{name} must divide TP")
        if self.request_batch < 1 or self.mtp_drafts < 0:
            raise ValueError("Invalid request/MTP geometry")

    @property
    def target_rows(self):
        return self.request_batch * (self.mtp_drafts + 1)

    @property
    def local(self):
        return {
            k: getattr(self, k) // self.tp
            for k in ("intermediate", "vocab", "q_heads", "kv_heads", "gdn_k_heads", "gdn_v_heads")
        }
