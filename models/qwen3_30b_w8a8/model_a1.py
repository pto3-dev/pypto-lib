# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Single-card resident 48-layer A1 composition with explicit checkpoint contracts."""

import json
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import torch

from .decode_a1 import A1DecodeMetadata, A1DecoderImages, A1DecoderLayer, compile_a1_decoder_images
from .vocab_a1 import compile_a1_lm_head, prepack_a1_lm_head
from .weights_a1 import load_a1_attention_weights, load_a1_expert_layer, load_a1_vocab_weights


def make_a1_rope_cache(
    checkpoint: Path, *, storage_dtype: torch.dtype = torch.bfloat16, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Build an unscaled NeoX table on the explicit device before storage casting.

    Native vLLM initializes this table on NPU. CPU construction is preserved as
    the historical BF16 default, not substituted for native FP16 initialization.
    """
    if storage_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"A1 RoPE requires BF16 or FP16 storage, got {storage_dtype}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("rope_scaling") is not None or config.get("head_dim") != 128:
        raise ValueError("A1 RoPE requires unscaled head_dim=128")
    theta = float(config["rope_theta"])
    frequencies = 1.0 / (theta ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128))
    phase = torch.einsum("i,j->ij", torch.arange(512, device=device, dtype=torch.float32), frequencies)
    return torch.cat((phase.cos(), phase.sin()), dim=1).to(storage_dtype)


class A1ResidentModel:
    """Own all 48 ND weight banks, caches, prepared launches and one NPU graph.

    The default checkpoint path remains BF16/W8A8. Explicit modelslim=True
    selects native FP16/static-Attention/dynamic-expert semantics and checked
    ModelSlim loading; it is not an implicit dtype conversion or fallback.
    Prefill can replay the same causal graph sequentially. This component
    alone is not a vLLM service or end-to-end precision/performance proof.
    """

    def __init__(
        self,
        checkpoint: Path,
        build_dir: Path,
        pto_isa_root: Path,
        *,
        device: torch.device,
        images: A1DecoderImages | None = None,
        progress: Callable[[int], None] | None = None,
        modelslim: bool = False,
    ) -> None:
        from simpler_setup import A1BorrowedLaunch, KernelCompiler

        if device.type != "npu" or device.index is None:
            raise ValueError(f"A1 resident model requires an indexed NPU device, got {device}")
        self.device = device
        self._owners = ExitStack()
        self._closed = False
        self.graph = None
        self._graph_has_logits = False
        self.layers = []
        self.modelslim = modelslim
        storage_dtype = torch.float16 if modelslim else torch.bfloat16
        loader = None
        if modelslim:
            from .decode_modelslim_a1 import A1ModelSlimDecoderLayer, compile_a1_modelslim_decoder_images
            from .weights_modelslim_a1 import A1ModelSlimWeights

            loader = A1ModelSlimWeights(checkpoint)
            self.images = images or compile_a1_modelslim_decoder_images(build_dir / "decoder", pto_isa_root)
        else:
            self.images = images or compile_a1_decoder_images(build_dir / "decoder", pto_isa_root)
        self.hidden = torch.zeros((49, 2048), dtype=storage_dtype, device=device)
        self.residual = torch.zeros_like(self.hidden)
        self.token = torch.zeros((1, 8), dtype=torch.int32, device=device)
        self.metadata = A1DecodeMetadata(
            torch.zeros((1, 8), dtype=torch.int32, device=device),
            torch.full((1, 8), -1, dtype=torch.int32, device=device),
            torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]], dtype=torch.int32, device=device),
            torch.zeros((1, 8), dtype=torch.int32, device=device),
            make_a1_rope_cache(
                checkpoint, storage_dtype=storage_dtype, device=device if modelslim else "cpu"
            ).to(device),
        )
        self.key_cache = torch.zeros((48, 512, 512), dtype=storage_dtype, device=device)
        self.value_cache = torch.zeros_like(self.key_cache)
        self.output = torch.zeros((1, 2048), dtype=storage_dtype, device=device)
        self.final_skip = torch.zeros_like(self.output)
        self.final_quant = torch.zeros((1, 2048), dtype=torch.int8, device=device)
        self.final_scale = torch.zeros((1, 16), dtype=torch.float32, device=device)
        self.logits = torch.zeros((1, 151936), dtype=torch.float32, device=device)
        self.status = {}
        try:
            vocab = loader.vocabulary() if loader is not None else load_a1_vocab_weights(checkpoint)
            self.embedding = vocab["model.embed_tokens.weight"].to(device)
            self.norm_weight = vocab["model.norm.weight"].reshape(1, 2048).to(device)
            if modelslim:
                self.norm_bias = vocab["model.norm.bias"].reshape(1, 2048).to(device)
                self.lm_head = vocab["lm_head.weight"].t().contiguous().to(device)
            else:
                self.lm_head = prepack_a1_lm_head(vocab["lm_head.weight"]).to(device)
            del vocab
            compiler = KernelCompiler("a1")
            embedding_image = compiler.compile_single_launch(
                str(Path(__file__).with_name("a1_embedding.cpp")), str(pto_isa_root)
            )

            def native(name: str, image: bytes, tensors: tuple[torch.Tensor, ...]):
                owner = self._owners.enter_context(A1BorrowedLaunch(image, device_id=device.index))
                self.status[name] = torch.full((8, 16), -7, dtype=torch.int32, device=device)
                return owner.prepare(*tensors, self.status[name], blocks=8)

            self._embedding_call = native(
                "embedding", embedding_image, (self.token, self.embedding, self.hidden[:1])
            )
            for index in range(48):
                if loader is not None:
                    layer = A1ModelSlimDecoderLayer(
                        self.images,
                        self.hidden[index : index + 1],
                        self.residual[index : index + 1],
                        self.metadata,
                        (self.key_cache[index], self.value_cache[index]),
                        loader.attention(index),
                        loader.experts(index),
                        self.hidden[index + 1 : index + 2],
                        self.residual[index + 1 : index + 2],
                        first_layer=index == 0,
                    )
                else:
                    layer = A1DecoderLayer(
                        self.images,
                        self.hidden[index : index + 1],
                        self.residual[index : index + 1],
                        self.metadata,
                        self.key_cache[index],
                        self.value_cache[index],
                        load_a1_attention_weights(checkpoint, index),
                        load_a1_expert_layer(checkpoint, index),
                        self.hidden[index + 1 : index + 2],
                        self.residual[index + 1 : index + 2],
                    )
                self._owners.enter_context(layer)
                self.layers.append(layer)
                if progress is not None:
                    progress(index + 1)
            norm = self.images.dsl["norm"]
            owner = self._owners.enter_context(A1BorrowedLaunch(norm.image, device_id=device.index))
            tensors = {
                "x": self.hidden[48:49],
                "residual": self.residual[48:49],
                "weight": self.norm_weight,
                "y": self.output,
                "residual_out": self.final_skip,
                "quant": self.final_quant,
                "scale": self.final_scale[:, :1],
            }
            if modelslim:
                tensors["bias"] = self.norm_bias
            self._norm_call = owner.prepare(*(tensors[name] for name in norm.order))
            if modelslim:
                from .compile_modelslim_a1 import compile_modelslim_source

                head_image = compile_modelslim_source(
                    "a1_modelslim_lm_head.cpp", build_dir / "lm_head", pto_isa_root
                )
            else:
                head_image = compile_a1_lm_head(pto_isa_root)
            self._lm_head_call = native("lm_head", head_image, (self.output, self.lm_head, self.logits))
        except BaseException:
            self.close()
            raise

    def enqueue_hidden(self) -> None:
        """Enqueue embedding, all decoder layers and final Norm."""
        if self._closed:
            raise RuntimeError("A1 resident model is closed")
        self.enqueue_embedding()
        for layer in self.layers:
            layer.enqueue()
        self._norm_call()

    def enqueue(self) -> None:
        """Enqueue embedding, all decoder layers, final Norm and LM Head."""
        self.enqueue_hidden()
        self.enqueue_lm_head()

    def enqueue_embedding(self) -> None:
        """Enqueue the prepared raw 16-bit embedding operator."""
        if self._closed:
            raise RuntimeError("A1 resident model is closed")
        self._embedding_call()

    def enqueue_lm_head(self) -> None:
        """Enqueue the prepared vocabulary projection over the final Norm output."""
        if self._closed:
            raise RuntimeError("A1 resident model is closed")
        self._lm_head_call()

    def capture(self, *, include_logits: bool = True) -> None:
        """Warm up once outside capture and capture all model computation."""
        if self.graph is not None:
            raise RuntimeError("A1 resident model graph is already captured")
        enqueue = self.enqueue if include_logits else self.enqueue_hidden
        enqueue()
        torch.npu.synchronize(self.device)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            enqueue()
        self._graph_has_logits = include_logits

    def replay_token(self, token: int, position: int) -> torch.Tensor:
        """Update fixed device carriers then replay one token's causal model graph."""
        if self.graph is None or self._closed:
            raise RuntimeError("A1 resident model requires a live captured graph")
        if not self._graph_has_logits:
            raise RuntimeError("replay_token requires a graph captured with include_logits=True")
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 151936:
            raise ValueError(f"Expected token integer in [0,151936), got {token}")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < 512:
            raise ValueError(f"Expected position integer in [0,512), got {position}")
        self.token.fill_(token)
        self.metadata.position.fill_(position)
        self.metadata.slot.fill_(position)
        self.metadata.lengths.fill_(position + 1)
        self.graph.replay()
        return self.logits

    def close(self) -> None:
        """Destroy the owning graph before prepared resources; never reset the device."""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self._owners.close()
        self._closed = True

    def __enter__(self) -> "A1ResidentModel":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
