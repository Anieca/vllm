# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GIT (Generative Image-to-text Transformer) model.

Paper: https://arxiv.org/abs/2205.14100
Weights: https://huggingface.co/models?other=git (e.g. `microsoft/git-base-coco`)

GIT feeds the projected CLIP patch features in front of the text tokens of a
BERT-style (post-LayerNorm) decoder. The image tokens attend to each other
bidirectionally, while text tokens attend causally to the image and to the
preceding text. This is the multimodal prefix-LM mask (`is_mm_prefix_lm`).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature, GitConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .module_mapping import MultiModelKeys
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


def get_git_num_image_tokens(config: GitConfig) -> int:
    """Number of CLIP tokens (patches + class token) fed per image."""
    vision_config = config.vision_config
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    return num_patches + 1


class GitImagePixelInputs(TensorSchema):
    """Dimensions:
    - bn: Batch size * number of images
    - c: Number of channels (3)
    - h: Height of each image
    - w: Width of each image
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]


class GitProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> GitConfig:
        return self.ctx.get_hf_config(GitConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        return get_git_num_image_tokens(self.get_hf_config())


class GitDummyInputsBuilder(BaseDummyInputsBuilder[GitProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        image_size = self.info.get_hf_config().vision_config.image_size

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class GitMultiModalProcessor(BaseMultiModalProcessor[GitProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _postprocess_prompt(self, prompt: list[int]) -> list[int]:
        # The BERT tokenizer wraps the prompt as `[CLS] ... [SEP]`, but GIT
        # generates the continuation of `[CLS] ...`; a trailing `[SEP]` (GIT's
        # EOS token) would end the caption before it starts.
        eos_token_id = self.info.get_hf_config().eos_token_id
        if prompt and prompt[-1] == eos_token_id:
            return prompt[:-1]
        return prompt

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        # GIT has no image token; any id works since the embeddings of the
        # placeholders are overwritten by the image features.
        image_tokens = [hf_config.pad_token_id] * self.info.get_num_image_tokens()

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=image_tokens,
            )
        ]


class GitEmbeddings(nn.Module):
    """Word + position embeddings of the text tokens, followed by LayerNorm.

    The image tokens occupy the first `num_image_tokens` positions of the
    sequence. They are passed through unchanged, and the text positions restart
    from 0 after them, matching `GitModel` in HF Transformers.
    """

    def __init__(self, config: GitConfig, prefix: str = "") -> None:
        super().__init__()
        self.num_image_tokens = get_git_num_image_tokens(config)
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.word_embeddings",
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor):
        is_text = (positions >= self.num_image_tokens).unsqueeze(-1)
        text_positions = (positions - self.num_image_tokens).clamp(min=0)
        text_embeds = self.LayerNorm(
            inputs_embeds + self.position_embeddings(text_positions)
        )
        return torch.where(is_text, text_embeds, inputs_embeds)


class GitSelfAttention(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tp_size == 0
        self.num_heads = total_num_heads // tp_size
        self.head_dim = self.hidden_size // total_num_heads
        self.q_size = self.num_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.head_dim**-0.5,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size] * 3, dim=-1)
        return self.attn(q, k, v)


class GitSelfOutput(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class GitAttention(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self = GitSelfAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self"
        )
        self.output = GitSelfOutput(config, quant_config, prefix=f"{prefix}.output")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(hidden_states), hidden_states)


class GitIntermediate(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.intermediate_act_fn(hidden_states)


class GitOutput(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class GitLayer(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = GitAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.attention"
        )
        self.intermediate = GitIntermediate(
            config, quant_config, prefix=f"{prefix}.intermediate"
        )
        self.output = GitOutput(config, quant_config, prefix=f"{prefix}.output")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class GitEncoder(nn.Module):
    def __init__(
        self,
        config: GitConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                GitLayer(
                    config, cache_config, quant_config, prefix=f"{prefix}.layer.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


@support_torch_compile
class GitTextModel(nn.Module):
    """The text decoder of GIT, i.e. `GitModel` without the image encoder."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embeddings = GitEmbeddings(config, prefix=f"{prefix}.embeddings")
        self.encoder = GitEncoder(
            config, cache_config, quant_config, prefix=f"{prefix}.encoder"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.embeddings(inputs_embeds, positions)
        return self.encoder(hidden_states)


class GitForTextGeneration(nn.Module):
    """Text decoder + LM head; the language model of `GitForCausalLM`."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.model = GitTextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)


@MULTIMODAL_REGISTRY.register_processor(
    GitMultiModalProcessor,
    info=GitProcessingInfo,
    dummy_inputs=GitDummyInputsBuilder,
)
class GitForCausalLM(nn.Module, SupportsMultiModal):
    """GIT for image captioning and VQA.

    The prompt must contain exactly one image: the image tokens are always the
    first `num_image_tokens` positions, which is how `GitEmbeddings` tells them
    apart from text tokens. Multi-frame (video) checkpoints that set
    `num_image_with_embedding` are not supported.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"embeddings.position_ids": None},
        orig_to_new_stacked={
            ".self.query.": (".self.qkv_proj.", "q"),
            ".self.key.": (".self.qkv_proj.", "k"),
            ".self.value.": (".self.qkv_proj.", "v"),
        },
        orig_to_new_prefix={
            "git.image_encoder.": "image_encoder.",
            "git.visual_projection.visual_projection.": "visual_projection.",
            "git.": "language_model.model.",
            "output.": "language_model.lm_head.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: GitConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if config.num_image_with_embedding is not None:
            raise NotImplementedError(
                "GIT checkpoints with `num_image_with_embedding` (video) "
                "are not supported yet."
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.image_encoder = CLIPVisionModel(
                config.vision_config,
                quant_config,
                # GIT applies the post LayerNorm to every token, not only to
                # the pooled class token.
                require_post_norm=True,
                prefix=maybe_prefix(prefix, "image_encoder"),
            )
            self.visual_projection = nn.Sequential(
                nn.Linear(config.vision_config.hidden_size, config.hidden_size),
                nn.LayerNorm(
                    config.hidden_size, eps=config.vision_config.layer_norm_eps
                ),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = GitForTextGeneration(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> GitImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        image_size = self.config.vision_config.image_size
        return GitImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            resolve_bindings={"h": image_size, "w": image_size},
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        image_features = self.image_encoder(image_input.data)
        return self.visual_projection(image_features)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual_projection",
            tower_model="image_encoder",
        )
