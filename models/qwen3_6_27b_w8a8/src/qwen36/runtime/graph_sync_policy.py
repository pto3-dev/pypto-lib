# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit graph sync guard, vendored once from the pinned vLLM-Ascend method.

No inspect/AST/exec during initialization. Only the CPU barrier guard differs;
all graph event and device dependencies match the pinned implementation.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from vllm_ascend.compilation import acl_graph
from qwen36.runtime import host_timeline


def guarded_graph_call(self, *args, **kwargs):
    forward_context = acl_graph.get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode
    if aclgraph_runtime_mode == acl_graph.CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)
    if batch_descriptor not in self.concrete_aclgraph_entries:
        self.concrete_aclgraph_entries[batch_descriptor] = acl_graph.ACLGraphEntry(
            batch_descriptor=batch_descriptor
        )
    entry = self.concrete_aclgraph_entries[batch_descriptor]
    if entry.aclgraph is None:
        if self.aclgraph_options.debug_log_enable:
            acl_graph.logger.debug(
                "Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor
            )
        acl_graph.validate_cudagraph_capturing_enabled()
        input_addresses = [x.data_ptr() for x in args if isinstance(x, acl_graph.torch.Tensor)]
        entry.input_addresses = input_addresses
        aclgraph = acl_graph.torch.npu.NPUGraph()
        with acl_graph.ExitStack() as stack:
            if self.aclgraph_options.gc_disable:
                stack.enter_context(acl_graph.patch("gc.collect", lambda: None))
                stack.enter_context(acl_graph.patch("torch.npu.empty_cache", lambda: None))
            forward_context.capturing = True
            with acl_graph.torch.npu.graph(
                aclgraph,
                pool=self.graph_pool,
                stream=getattr(self.runnable, "_qwen38_mpmd_capture_stream", None),
            ):
                output = self.runnable(*args, **kwargs)
                if self.aclgraph_options.weak_ref_output:
                    output = acl_graph.weak_ref_tensors(output)
        global _graph_params
        global _draft_graph_params
        acl_graph.weak_ref_workspaces(acl_graph._graph_params)
        acl_graph.weak_ref_workspaces(acl_graph._draft_graph_params)
        entry.output = acl_graph.weak_ref_tensors(output)
        entry.aclgraph = aclgraph
        acl_graph.compilation_counter.num_cudagraph_captured += 1
        return output
    if self.is_debugging_mode:
        new_input_addresses = [x.data_ptr() for x in args if isinstance(x, acl_graph.torch.Tensor)]
        assert new_input_addresses == entry.input_addresses, (
            f"Input addresses for aclgraphs are different during replay. Expected {entry.input_addresses}, got {new_input_addresses}"
        )
    acl_graph.logger.info_once("Replaying aclgraph")
    is_draft_eagle = acl_graph._EXTRA_CTX.is_draft_model and self.use_eagle
    if (not self.enable_enpu and (not is_draft_eagle)) and (
        not getattr(forward_context, "qwen38_pto_skip_graph_sync", False)
    ):
        if host_timeline.ENABLED:
            timing = host_timeline.begin()
            acl_graph.torch.npu.current_stream().synchronize()
            host_timeline.end("graph_pre_replay_sync", timing)
        else:
            acl_graph.torch.npu.current_stream().synchronize()
    entry.aclgraph.replay()
    return entry.output


def install_graph_sync_guard(cls):
    expected = "d57263420a58484d510d5b3cb5aa9f57c13b88b7dbd9a3ded13bc40fc7177110"
    if hashlib.sha256(Path(acl_graph.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Pinned ACLGraphWrapper changed; review the explicit guard source")
    original = cls.__call__
    cls.__call__ = guarded_graph_call
    return original
