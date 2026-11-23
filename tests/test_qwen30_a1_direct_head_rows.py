"""Direct PTO head row loads preserve both native Norm and RoPE exactly."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("heads", (4, 32))
def test_native_head_row_loads_graph(tmp_path, heads):
    """Direct vector loads retain both native head Norm and RoPE values."""

    import pypto.language as pl
    import torch_npu
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.qk_norm_rope_batched_a1 import make_a1_batched_head_norm_rope
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    padded = max(heads, 8)
    kind = 'q' if heads == 32 else 'k'
    checkpoint = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    loaded = A1ModelSlimWeights(Path(checkpoint)).attention(0)
    gamma = loaded[f'self_attn.{kind}_norm.weight'].reshape(1, 128).npu()
    x = torch.zeros((20 * padded, 128), dtype=torch.float16, device='npu:0')
    rope = torch.zeros((20, 128), dtype=torch.float16, device='npu:0')
    outputs = [[torch.zeros_like(x), torch.zeros_like(x)] for _ in range(2)]
    generator = torch.Generator().manual_seed(3103202 + heads)
    old = compile_modelslim_program(
        make_a1_batched_head_norm_rope(heads, rows=20, storage_dtype=pl.FP16), tmp_path / 'control', Path(isa))
    model = Path(__file__).parents[1] / 'models/qwen3_30b_w8a8'
    source = model / f'a1_head_norm_rope_direct_{kind}.cpp'
    candidate = KernelCompiler('a1').compile_single_launch(str(source), isa,
                                                          extra_include_dirs=[str(model)])
    images = [old, (candidate, ('x', 'weight', 'cos_sin', 'norm', 'rotated'))]
    with ExitStack() as owners:
        graphs = []
        for arm, (image, order) in enumerate(images):
            owner = owners.enter_context(A1BorrowedLaunch(image))
            args = {'x': x, 'weight': gamma, 'cos_sin': rope,
                    'norm': outputs[arm][0], 'rotated': outputs[arm][1]}
            call = owner.prepare(*(args[label] for label in order), blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                for _ in range(32):
                    call()
            graphs.append(graph)
        for case in range(4):
            values = torch.randn((20 * padded, 128), generator=generator).half()
            if case == 0:
                values.zero_()
            x.copy_(values)
            phase = torch.randn((20, 64), generator=generator) * (case + 1)
            rope.copy_(torch.cat((phase.cos(), phase.sin()), dim=-1).half())
            norm, _ = torch_npu.npu_rms_norm(x.reshape(20, padded, 128)[:, :heads].contiguous(),
                                            gamma.reshape(128), 1e-6)
            cosine = rope[:, :64].repeat(1, 2).reshape(20, 1, 1, 128)
            sine = rope[:, 64:].repeat(1, 2).reshape(20, 1, 1, 128)
            rotated, _ = torch_npu.npu_apply_rotary_pos_emb(
                norm.clone().reshape(20, 1, heads, 128), norm.clone().reshape(20, 1, heads, 128),
                cosine, sine, rotary_mode='half')
            for graph in graphs:
                graph.replay()
            for index, expected in enumerate((norm, rotated.reshape(20, heads, 128))):
                assert torch.equal(outputs[0][index].cpu(), outputs[1][index].cpu()), (heads, case, index)
                actual = outputs[1][index].reshape(20, padded, 128)[:, :heads].cpu()
                assert torch.equal(actual, expected.cpu()), (heads, case, index, 'native')
        times = []
        for graph in graphs:
            for _ in range(5):
                graph.replay()
            samples = []
            for _ in range(30):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) / 32.0)
            times.append(samples)
        (tmp_path / 'direct-rope-times.json').write_text(json.dumps({
            'heads': heads, 'native_exact': True, 'captured_repetitions': 32, 'control_ms': times[0], 'direct_ms': times[1],
            'control_median_ms': statistics.median(times[0]),
            'candidate_median_ms': statistics.median(times[1])}, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
