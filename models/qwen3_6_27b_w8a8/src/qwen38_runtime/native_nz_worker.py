# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .mtp_async_merged_graph import install as install_mtp_graph, stats as mtp_graph_stats

install_mtp_graph()
from .worker_lifecycle import MPMDBenchmarkWorker


class NativeNzWorker(MPMDBenchmarkWorker):
    def benchmark_mpmd_stats(self):
        result = MPMDBenchmarkWorker.benchmark_mpmd_stats(self)
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

        builders = []
        for groups in self.model_runner.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(0)
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    builders.append(
                        dict(
                            raw_builds=getattr(builder, "_pto_raw_builds", 0),
                            native_builds=getattr(builder, "_pto_native_builds", 0),
                            validated_builds=getattr(builder, "_pto_validated_builds", 0),
                            cache_mode=builder.vllm_config.cache_config.mamba_cache_mode,
                            source_dtypes=getattr(builder, "_pto_raw_source_dtypes", {}),
                            rejection_reasons=getattr(builder, "_pto_rejection_reasons", {}),
                        )
                    )
        result["gdn_metadata_builders"] = builders
        drafter = getattr(self.model_runner, "drafter", None)
        result["mtp_async_graph"] = mtp_graph_stats(drafter)
        result["native_drafter"] = dict(
            use_cuda_graph=getattr(drafter, "use_cuda_graph", None),
            async_scheduling=getattr(drafter, "use_async_scheduling", None),
            model_type=type(getattr(drafter, "model", None)).__name__,
        )
        return result
