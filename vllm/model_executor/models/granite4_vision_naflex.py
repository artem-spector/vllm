# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM implementation of Granite Vision 5 (granite4_vision_naflex).

Built on top of granite4_vision (Granite 4.1 Vision, already built-in) --
adds NaFlex native-resolution vision (packed variable-patch-count images via
spatial_shapes, replacing granite4_vision's fixed-tile anyres scheme) and
image M-RoPE (3-axis (t, h, w) multimodal rotary position embeddings for
injected image tokens, absent in granite4_vision).

Deliberately NOT sharing code with granite4_vision.py -- the two models'
vision/position pipelines diverge enough (fixed-tile anyres vs. NaFlex
native-resolution, no M-RoPE vs. 3-axis M-RoPE) that coupling them would cost
more than the duplication of the parts that stay the same (the DeepStack
buffer-injection scaffolding).

Architectural summary (see gv4_release/models/gv5.md for the full writeup):
  - SigLIP2 NaFlex vision tower: packed variable-resolution patches
    (spatial_shapes per image), plain full attention (NOT windowed/blocked --
    all patches from all images in a batch attend to each other freely;
    there is no per-image attention mask), learned absolute position
    embedding resized per image, optional 2D vision RoPE (config rope_scale,
    off for the checkpoints seen so far -- round_robin-interleaved row/col,
    wired but unused).
  - NaflexWindowQFormerDownsampler: per-image variable-rectangle
    window-qformer, one instance per deepstack level + one per spatial-offset
    group, each owning its own learnable newline / optional 2D position
    embedding (no shared model-level image_newline).
  - DeepStack + spatial-offset injection into the Granite LLM backbone --
    same scaffolding granite4_vision.py uses (buffer-per-level, ds_{layer}
    IntermediateTensors keys).
  - Image M-RoPE (3-axis (t, h, w) for checkpoints seen so far): text tokens
    carry h==w==t==running scalar (rotary collapses to 1D); image tokens
    carry per-view (row, col) with an optional band_center recentering, plus
    a per-image-constant t (the image's timeline centroid). Channel
    allocation is round_robin over (h, w) in the HIGH-frequency channels,
    with a contiguous LOW-frequency block reserved entirely for t
    (image_mrope_t_channels) -- this does NOT match vLLM's existing
    MRotaryEmbedding(mrope_interleaved=True) (Qwen3-VL's period-3
    [T H T H W H ...] interleave) or XDRotaryEmbedding (contiguous split, no
    t-carve-out), so a small custom rotary class ports the model card's own
    mrope.py:apply_mrope verbatim.

hf_overrides requirement: vLLM's M-RoPE activation (ModelConfig.uses_mrope)
is a config-driven check (whether hf_config.get_text_config().rope_parameters
contains an "mrope_section" key) that is completely independent of whether
the model class implements SupportsMRoPE. Checkpoints of this architecture
whose text_config has no "mrope_section" key (the common case, since this is
not a native transformers/vLLM rope_parameters convention) MUST be loaded
with `hf_overrides=granite4_vision_naflex.hf_overrides` passed to `LLM(...)`
or the serving engine args -- otherwise vLLM silently never calls
get_mrope_input_positions and falls back to plain 1D positions. See
hf_overrides()'s own docstring below for why this can't be done inside the
model class's __init__ instead.
"""

import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.blip2 import Blip2QFormerModel
from vllm.model_executor.models.granite import GraniteForCausalLM, GraniteModel
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.llava import LlavaDummyInputsBuilder
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig
from vllm.multimodal.parse import ImageProcessorItems, ImageSize
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def hf_overrides(config):
    """REQUIRED: pass as `hf_overrides=granite4_vision_naflex.hf_overrides` to
    `LLM(...)` / the serving engine args -- the model class alone is not
    enough to activate vLLM's M-RoPE code path.

    vLLM's `ModelConfig.uses_mrope` (vllm/config/model.py) is a config-driven
    property, NOT tied to whether the model class implements SupportsMRoPE --
    it purely checks whether `hf_config.get_text_config().rope_parameters`
    contains an `mrope_section` key (vllm/transformers_utils/config.py). A
    checkpoint whose text_config carries a plain rope_parameters dict (just
    rope_theta/rope_type, no mrope_section) makes `uses_mrope` evaluate
    False, so the engine never allocates the [3, N] mrope positions buffer
    and get_mrope_input_positions is never called at all (silently -- no
    error).

    This must happen via `hf_overrides` (applied inside `ModelConfig.__init__`,
    before the model class is ever constructed) rather than inside
    Granite4VisionNaflexForConditionalGeneration.__init__ -- by the time the
    model class runs, GPUModelRunner.__init__ has already read and cached
    model_config.uses_mrope, so a mutation inside the model's own __init__ is
    too late.

    The actual channel-allocation values (image_mrope_t_channels,
    image_mrope_method) are read directly by
    Granite4VisionNaflexMRotaryEmbedding from text_config/top-level config --
    the placeholder mrope_section planted here is never consumed by our own
    rotary class's channel-selection math. BUT its VALUES still matter, not
    just its presence: get_rope() (called from GraniteAttention.__init__,
    before our rotary swap runs) constructs a throwaway MRotaryEmbedding
    first, and that constructor asserts sum(mrope_section) == rotary_dim // 2.
    The placeholder must sum to exactly head_dim // 2.
    """
    if not getattr(config, "use_image_mrope", False):
        return config
    text_config = config.get_text_config()
    rope_params = dict(getattr(text_config, "rope_parameters", None) or {})
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads
    )
    half = head_dim // 2
    # Values are irrelevant to our own rotary class (see docstring) -- only
    # the sum needs to satisfy MRotaryEmbedding's throwaway-construction
    # assert. Split roughly evenly across 3 sections; any split summing to
    # `half` works.
    placeholder_section = [half // 3, half // 3, half - 2 * (half // 3)]
    rope_params.setdefault("mrope_section", placeholder_section)
    text_config.rope_parameters = rope_params
    return config


# ---------------------------------------------------------------------------
# SigLIP2 NaFlex vision tower (packed variable-resolution patches)
#
# NOT based on vllm's siglip2navit.py: that file is built for Qwen-style
# (t, h, w) video grids with a window-attention hierarchy (hidden_stride,
# get_window_index, fullatt_block_indexes) -- none of which this model has.
# This tower is a plain full-attention encoder over packed (H, W)-only
# patches, matching vanilla HF Siglip2 with two local additions: absolute PE
# resize (present in upstream Siglip2 too) and an optional, currently-off,
# 2D vision RoPE.
# ---------------------------------------------------------------------------


class Siglip2NaflexVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            config.num_channels * self.patch_size * self.patch_size, self.embed_dim
        )
        self.num_patches = getattr(config, "num_patches", None) or (
            config.image_size // self.patch_size
        ) ** 2
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """positional_embeddings: (side, side, C). spatial_shapes: (num_images, 2) = (H, W).
        Returns (1, total_patches, C), packed to match pixel_values layout."""
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        pe = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if pe.device.type == "cpu":
            pe = pe.to(torch.float32)

        all_embeddings = []
        for i in range(spatial_shapes.shape[0]):
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(
                pe, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1).to(source_dtype)
            all_embeddings.append(resized)
        return torch.cat(all_embeddings, dim=0).unsqueeze(0)

    def forward(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
    ) -> torch.Tensor:
        """pixel_values: (1, total_patches, C*patch*patch), packed. spatial_shapes: (n_images, 2)."""
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_pe = self.resize_positional_embeddings(positional_embeddings, spatial_shapes)
        return patch_embeds + resized_pe


class Siglip2NaflexVisionRotaryEmbedding(nn.Module):
    """2D vision RoPE frequency table. Wired for parity with the model
    card's rope_scale knob, currently unused (rope_scale == 0.0 -> identity
    for checkpoints seen so far)."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_vision_rope(q, k, cos, sin, scale: float):
    """q, k: (1, num_heads, seq, head_dim). cos, sin: (seq, head_dim). scale==0.0 -> identity."""
    if scale == 0.0:
        return q, k
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    if scale == 1.0:
        return q_rot, k_rot
    return (1.0 - scale) * q + scale * q_rot, (1.0 - scale) * k + scale * k_rot


class Siglip2NaflexAttention(nn.Module):
    """Plain full attention over the packed (batch=1) sequence -- no
    per-image blocking. The HF reference forward() never constructs or
    passes an attention_mask, so every patch from every image in the packed
    batch attends to every other patch freely."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_rope = getattr(config, "use_rope", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, embed_dim = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            rope_scale = getattr(self.config, "rope_scale", 0.0)
            q, k = _apply_vision_rope(q, k, cos, sin, scale=rope_scale)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)


class Siglip2NaflexMLP(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class Siglip2NaflexEncoderLayer(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2NaflexAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2NaflexMLP(config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Siglip2NaflexEncoder(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Siglip2NaflexEncoderLayer(
                    config, quant_config=quant_config, prefix=f"{prefix}.layers.{idx}"
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states = inputs_embeds
        all_hidden_states = [hidden_states] if return_all_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)
        return all_hidden_states if return_all_hidden_states else hidden_states


class Siglip2NaflexVisionModel(nn.Module):
    """Top-level NaFlex SigLIP2 vision tower. embeddings + encoder +
    post_layernorm, matching the checkpoint's vision_model.* weight-key
    layout."""

    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        head_dim = embed_dim // config.num_attention_heads

        self.vision_model = nn.Module()
        self.vision_model.embeddings = Siglip2NaflexVisionEmbeddings(config)
        self.vision_model.encoder = Siglip2NaflexEncoder(
            config, quant_config=quant_config, prefix=f"{prefix}.vision_model.encoder"
        )
        self.vision_model.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        if head_dim % 4 != 0:
            raise ValueError(
                f"2D RoPE needs head_dim divisible by 4, got head_dim={head_dim}"
            )
        self.rotary_emb = Siglip2NaflexVisionRotaryEmbedding(head_dim // 2)

    def _compute_2d_rope(self, spatial_shapes: torch.Tensor, device, dtype):
        shapes = spatial_shapes.tolist()
        all_pos_ids = []
        for h, w in shapes:
            row_ids = torch.arange(h).unsqueeze(1).expand(h, w).reshape(-1)
            col_ids = torch.arange(w).unsqueeze(0).expand(h, w).reshape(-1)
            all_pos_ids.append(torch.stack([row_ids, col_ids], dim=-1))
        pos_ids = torch.cat(all_pos_ids, dim=0)

        max_grid = max((max(h, w) for h, w in shapes), default=1)
        max_grid = max(max_grid, 1)
        freq_table = self.rotary_emb(max_grid)
        emb = freq_table[pos_ids.to(freq_table.device)]
        emb = emb.transpose(1, 2).reshape(emb.shape[0], -1)
        emb = torch.cat([emb, emb], dim=-1)
        return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        spatial_shapes = spatial_shapes.cpu()
        hidden_states = self.vision_model.embeddings(pixel_values, spatial_shapes)

        use_rope = getattr(self.config, "use_rope", False)
        position_embeddings = (
            self._compute_2d_rope(spatial_shapes, hidden_states.device, hidden_states.dtype)
            if use_rope
            else None
        )

        encoder_out = self.vision_model.encoder(
            hidden_states,
            position_embeddings=position_embeddings,
            return_all_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            encoder_out[-1] = self.vision_model.post_layernorm(encoder_out[-1])
            return encoder_out

        return self.vision_model.post_layernorm(encoder_out)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ---------------------------------------------------------------------------
# Downsampler: NaFlex window-qformer, variable rectangle per image
# ---------------------------------------------------------------------------


class NaflexWindowQFormerDownsampler(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        spatial_offset: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        llm_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.dropout = nn.Dropout(config.projector_dropout)
        self._spatial_offset = spatial_offset
        if spatial_offset is not None:
            self.offset_h, self.offset_w = [(0, 0), (0, 1), (1, 0), (1, 1)][spatial_offset]

        qformer_config = Blip2QFormerConfig(
            hidden_size=vision_hidden_size,
            num_attention_heads=vision_hidden_size // 64,
            intermediate_size=3072,
            num_hidden_layers=1,
            encoder_hidden_size=vision_hidden_size,
            cross_attention_frequency=1,
            max_position_embeddings=2048,
            use_qformer_text_input=False,
        )
        self.qformer = Blip2QFormerModel(
            qformer_config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.qformer",
        )

        q, w = config.downsample_rate.split("/")
        self.query_side, self.window_side = int(q), int(w)
        self.query_length = self.query_side**2

        embed_std = 1 / math.sqrt(vision_hidden_size)
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.query = nn.Parameter(
            torch.randn(1, self.query_length, vision_hidden_size) * embed_std
        )
        self.image_positions = nn.Parameter(
            torch.randn(1, self.window_side**2, vision_hidden_size) * embed_std
        )
        self.out_linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

        llm_embed_std = 1 / math.sqrt(llm_hidden_size)
        self.pos_embed = None
        if getattr(config, "use_projector_pos_embed", False):
            g = config.projector_pos_embed_grid_size
            self.pos_embed_grid_size = g
            self.pos_embed = nn.Parameter(torch.randn(g * g, llm_hidden_size) * llm_embed_std)

        self.newline = None
        if getattr(config, "use_image_newline", False):
            self.newline = nn.Parameter(torch.randn(llm_hidden_size) * llm_embed_std)

    def _add_pos_embed(self, out_2d: torch.Tensor) -> torch.Tensor:
        new_H, new_W, C = out_2d.shape
        g = self.pos_embed_grid_size
        pe = self.pos_embed.reshape(g, g, C).permute(2, 0, 1).unsqueeze(0)
        if pe.device.type == "cpu":
            pe = pe.float()
        pe = F.interpolate(pe, size=(new_H, new_W), mode="bilinear", align_corners=False, antialias=True)
        pe = pe.squeeze(0).permute(1, 2, 0).to(out_2d.dtype)
        return out_2d + pe

    def _append_newline(self, out_2d: torch.Tensor) -> torch.Tensor:
        new_H, new_W, C = out_2d.shape
        col = self.newline.to(out_2d.dtype)[None, None, :].expand(new_H, 1, C)
        return torch.cat([out_2d, col], dim=1).reshape(new_H * (new_W + 1), C)

    def _win_rect(self, x: torch.Tensor, H: int, W: int, win: int) -> torch.Tensor:
        B, _, C = x.shape
        n_h, n_w = H // win, W // win
        return (
            x.view(B, H, W, C)
            .view(B, n_h, win, n_w, win, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B * n_h * n_w, win * win, C)
        )

    def _unwin_rect(self, xw: torch.Tensor, n_h: int, n_w: int, win: int) -> torch.Tensor:
        Bnn, _, C = xw.shape
        total_windows = n_h * n_w
        B = Bnn // total_windows
        H, W = n_h * win, n_w * win
        return (
            xw.view(B, n_h, n_w, win, win, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H, W, C)
            .flatten(1, 2)
        )

    def _spatial_offset_sample(self, feat: torch.Tensor, H: int, W: int):
        C = feat.shape[-1]
        blocks = feat.view(1, H, W, C).reshape(1, H // 2, 2, W // 2, 2, C)
        sampled = blocks[:, :, self.offset_h, :, self.offset_w, :]
        return sampled.reshape(1, -1, C), H // 2, W // 2

    def _interpolate_adaptive(self, feat: torch.Tensor, H: int, W: int):
        C = feat.shape[-1]
        n_h, n_w = H // self.window_side, W // self.window_side
        new_H, new_W = n_h * self.query_side, n_w * self.query_side
        feat_2d = feat.view(1, H, W, C).permute(0, 3, 1, 2)
        resized = F.interpolate(feat_2d, size=(new_H, new_W), mode="area")
        return resized.permute(0, 2, 3, 1).flatten(1, 2), new_H, new_W

    def _downsample(self, feat: torch.Tensor, H: int, W: int):
        if self._spatial_offset is not None:
            return self._spatial_offset_sample(feat, H, W)
        return self._interpolate_adaptive(feat, H, W)

    def forward(
        self, per_image_features: list[torch.Tensor], spatial_shapes: torch.Tensor
    ) -> list[torch.Tensor]:
        """per_image_features: list of (H_i*W_i, C), one per image. spatial_shapes: (B, 2)."""
        all_enc_windows = []
        all_query_windows = []
        window_counts = []
        output_shapes = []

        for img_idx, img_feat in enumerate(per_image_features):
            H, W = spatial_shapes[img_idx].tolist()
            feat = self.norm(img_feat.unsqueeze(0))

            enc = self._win_rect(feat, H, W, self.window_side)
            ds, new_H, new_W = self._downsample(feat, H, W)
            query = self._win_rect(ds, new_H, new_W, self.query_side)

            n_h, n_w = H // self.window_side, W // self.window_side
            all_enc_windows.append(enc)
            all_query_windows.append(query)
            window_counts.append(n_h * n_w)
            output_shapes.append((new_H, new_W))

        all_enc = torch.cat(all_enc_windows, dim=0)
        all_query = torch.cat(all_query_windows, dim=0)

        query_embeds = self.query + all_query
        encoder_embeds = self.dropout(all_enc + self.image_positions)
        out_windows = self.qformer(
            query_embeds=query_embeds, encoder_hidden_states=encoder_embeds
        )

        results = []
        offset = 0
        for n_win, (new_H, new_W) in zip(window_counts, output_shapes):
            img_out = out_windows[offset : offset + n_win]
            n_h, n_w = new_H // self.query_side, new_W // self.query_side
            flat = self._unwin_rect(img_out, n_h, n_w, self.query_side)
            out = self.out_linear(self.dropout(flat.squeeze(0)))

            if self.pos_embed is not None or self.newline is not None:
                out_2d = out.view(new_H, new_W, -1)
                if self.pos_embed is not None:
                    out_2d = self._add_pos_embed(out_2d)
                if self.newline is not None:
                    out = self._append_newline(out_2d)
                else:
                    out = out_2d.reshape(new_H * new_W, -1)

            results.append(out)
            offset += n_win

        return results


# ---------------------------------------------------------------------------
# Custom M-RoPE rotary embedding for the Granite text backbone
#
# Ports the model card's mrope.py:apply_mrope verbatim. NOT vllm's
# MRotaryEmbedding(mrope_interleaved=True): that produces a period-3
# [T H T H W H ...] interleave across the full channel range. This model
# instead carves a CONTIGUOUS block of the lowest-frequency channels
# exclusively for t (image_mrope_t_channels) and round-robins h/w only in
# the remaining higher-frequency channels -- t is never interleaved with
# h/w. No existing vLLM rotary layer implements this scheme.
# ---------------------------------------------------------------------------


def mrope_scalar_plane_index(n_axes: int) -> int:
    """Which plane of an [n_axes, ...] M-RoPE tensor is the scalar (timeline)
    plane -- the one a caller should treat as "the" position for
    causal-mask / boundary purposes. 2 axes -> (h, w) -> index 0. 3 axes ->
    (t, h, w) -> index 1 (t is a fractional per-image constant, unusable as
    a boundary signal)."""
    if n_axes == 2:
        return 0
    if n_axes == 3:
        return 1
    raise ValueError(f"unsupported mrope n_axes: {n_axes!r} (expected 2 or 3)")


class Granite4VisionNaflexMRotaryEmbedding(RotaryEmbeddingBase):
    """3-axis (t, h, w) M-RoPE for the Granite text backbone, matching the
    model card's mrope.py:apply_mrope(method="round_robin") channel
    allocation exactly."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        t_channels: int | None = None,
        method: str = "round_robin",
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)
        half = rotary_dim // 2
        self.t_channels = t_channels if t_channels is not None else rotary_dim // 8
        if not 0 < self.t_channels < half:
            raise ValueError(
                f"mrope t_channels must be in (0, {half}) for rotary_dim {rotary_dim}; "
                f"got {self.t_channels}"
            )
        hw_len = half - self.t_channels
        if hw_len % 2:
            raise ValueError(
                f"mrope t_channels leaves an odd h/w remainder ({hw_len}) for rotary_dim "
                f"{rotary_dim}; pick a t_channels with the same parity as rotary_dim/2"
            )
        if method not in ("round_robin", "split"):
            raise ValueError(f"unknown mrope method: {method!r}")
        self.method = method

        i = torch.arange(half)
        is_t = i >= hw_len  # trailing (lowest-frequency) block -> t
        if method == "round_robin":
            use_h = (i % 2) == 0
        else:
            use_h = i < (hw_len // 2)
        # vllm's ApplyRotaryEmb (rotary_embedding/common.py) expects cos/sin
        # at HALF width ([seq_len, head_size // 2]) -- it does its own
        # internal duplication via cos.unsqueeze(-2) broadcast across both
        # rotate_half chunks. Do NOT mirror these onto a full-width buffer
        # the way the model card's mrope.py:apply_mrope does (that
        # convention is for a DIFFERENT rotary application function that
        # consumes already-duplicated full-width cos/sin).
        self.register_buffer("_use_h", use_h, persistent=False)
        self.register_buffer("_is_t", is_t, persistent=False)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """positions: [3, num_tokens] (t, h, w). query/key: [num_tokens, num_heads*head_size].

        Always [3, num_tokens]: once a model declares SupportsMRoPE and
        model_config.uses_mrope is True, vLLM's gpu_model_runner always
        returns the [3, N] mrope_positions buffer for every request
        (text-only included -- get_mrope_input_positions's text-only
        fallback fills all 3 rows identically), never a bare 1D tensor."""
        assert positions.ndim == 2 and positions.shape[0] == 3
        assert key is not None
        num_tokens = positions.shape[-1]

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        cos_sin = cos_sin_cache[positions]  # [3, num_tokens, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)  # each [3, num_tokens, rotary_dim/2] (half width)

        # cos[0]/cos_h/cos_w etc. are 2D (num_tokens, half) -- one less dim
        # than `cos` itself (3D, [3, num_tokens, half]) since indexing
        # [0]/[1]/[2] already dropped the axis dim. view(1, -1) (not
        # (1, 1, -1)) matches that 2D shape -- an extra leading 1 here would
        # broadcast torch.where's OUTPUT to 3D, which propagates through
        # _apply_rope into query_rot, crashing the final torch.cat with
        # query_pass (still plain 3D) under torch.compile's fake-tensor
        # tracing (4D-vs-3D dimension mismatch).
        use_h = self._use_h.to(cos.device).view(1, -1)
        is_t = self._is_t.to(cos.device).view(1, -1)
        cos_t, cos_h, cos_w = cos[0], cos[1], cos[2]
        sin_t, sin_h, sin_w = sin[0], sin[1], sin[2]
        cos_sel = torch.where(use_h, cos_h, cos_w)
        sin_sel = torch.where(use_h, sin_h, sin_w)
        cos_sel = torch.where(is_t, cos_t, cos_sel)
        sin_sel = torch.where(is_t, sin_t, sin_sel)

        # Inline neox-style rotation instead of going through ApplyRotaryEmb
        # (a @CustomOp.register-ed class) -- calling it under torch.compile's
        # fake-tensor tracing produced a stray leading batch dim of 1 that
        # the non-rotated `_pass` slice (still 3D) didn't have, crashing the
        # subsequent torch.cat with a 4D-vs-3D mismatch; a plain inline
        # implementation has no such indirection to go wrong. Granite uses
        # neox-style rotation (GraniteAttention has no is_neox_style override
        # => get_rope's default True), so no GPT-J-style branch is needed
        # here.
        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        def _apply_rope(x_rot: torch.Tensor) -> torch.Tensor:
            cos_full = torch.cat((cos_sel, cos_sel), dim=-1).unsqueeze(-2).to(x_rot.dtype)
            sin_full = torch.cat((sin_sel, sin_sel), dim=-1).unsqueeze(-2).to(x_rot.dtype)
            return x_rot * cos_full + _rotate_half(x_rot) * sin_full

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rope(query_rot)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rope(key_rot)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    forward_cuda = forward_native

    @staticmethod
    def get_next_input_positions_tensor(out, out_offset, context_len, num_new_tokens):
        """Decode continuation: all 3 axes advance in lockstep with the same
        arange -- matches the text-token invariant (t == h == w == running
        scalar)."""
        import numpy as np

        values = np.arange(context_len, context_len + num_new_tokens, dtype=out.dtype)
        out[:, out_offset : out_offset + num_new_tokens] = values


# ---------------------------------------------------------------------------
# Granite text backbone with DeepStack injection + M-RoPE rotary override
# ---------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is [seq] for plain models but [3, seq] for M-RoPE (this
        # model, always -- use_image_mrope is a fixed architectural
        # property, never toggled per-request) -- dim 0 is a constant 3, NOT
        # the dynamic sequence dim. Must mark the LAST dim as dynamic
        # instead (matches Qwen3LLMModel's own dynamic_arg_dims for the
        # identical M-RoPE situation). GraniteModel's own
        # @support_torch_compile (no dynamic_arg_dims override) defaults to
        # dim 0 for every arg, which is wrong here.
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Granite4VisionNaflexLLMModel(GraniteModel):
    """GraniteModel with DeepStack feature injection + M-RoPE rotary override.

    The M-RoPE rotary swap (Granite4VisionNaflexMRotaryEmbedding replacing
    GraniteAttention's default rotary_emb) is done by monkeypatching each
    layer's self_attn.rotary_emb after super().__init__() runs, rather than
    subclassing GraniteAttention -- avoids duplicating its whole __init__
    (qkv/o_proj construction, TP sharding) just to change one line.

    mrope_config is passed explicitly (not read from
    vllm_config.model_config.hf_config) because by the time this class is
    constructed, vllm_config has already been swapped via
    with_hf_config(text_config) one level up (in
    Granite4VisionNaflexLLMForCausalLM) -- its hf_config IS the Granite text
    config, so use_image_mrope / image_mrope_axes / image_mrope_t_channels /
    image_mrope_method (which live on the TOP-LEVEL
    Granite4VisionNaflexConfig, not text_config) are no longer reachable
    from here."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mrope_config: dict | None = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config  # already the Granite text config here
        if mrope_config is not None:
            n_axes = mrope_config.get("axes", 2)
            if n_axes != 3:
                raise NotImplementedError(
                    "granite4_vision_naflex vLLM support only implements 3-axis "
                    f"M-RoPE, got image_mrope_axes={n_axes}"
                )
            t_channels = mrope_config.get("t_channels")
            method = mrope_config.get("method", "round_robin")
            rope_params = config.rope_parameters or {}
            base = rope_params.get("rope_theta", 10000.0)
            for layer in self.layers:
                attn = layer.self_attn
                mrope_emb = Granite4VisionNaflexMRotaryEmbedding(
                    attn.rotary_emb.head_size,
                    attn.rotary_emb.rotary_dim,
                    attn.rotary_emb.max_position_embeddings,
                    base,
                    attn.rotary_emb.is_neox_style,
                    attn.rotary_emb.dtype,
                    t_channels=t_channels,
                    method=method,
                ).to(next(attn.parameters()).device)
                attn.rotary_emb = mrope_emb

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
                hidden_states = hidden_states * self.config.embedding_multiplier
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if deepstack_input_embeds is None:
                ds_keys = [k for k in intermediate_tensors.tensors if k.startswith("ds_")]
                if ds_keys:
                    deepstack_input_embeds = IntermediateTensors(
                        {k: intermediate_tensors[k] for k in ds_keys}
                    )

        for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer_idx += self.start_layer
            if deepstack_input_embeds is not None:
                key = f"ds_{layer_idx}"
                if key in deepstack_input_embeds.tensors:
                    feat = deepstack_input_embeds[key]
                    num_tokens = hidden_states.size(0)
                    buf_len = feat.shape[0]
                    if buf_len != num_tokens:
                        feat = torch.nn.functional.pad(
                            feat[:num_tokens], (0, 0, 0, max(0, num_tokens - buf_len))
                        )
                    hidden_states = hidden_states + feat
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            remaining = (
                {
                    k: v
                    for k, v in deepstack_input_embeds.tensors.items()
                    if int(k.split("_")[1]) >= self.end_layer
                }
                if deepstack_input_embeds is not None
                else {}
            )
            return IntermediateTensors({"hidden_states": hidden_states, **remaining})

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Granite4VisionNaflexLLMForCausalLM(GraniteForCausalLM):
    """GraniteForCausalLM backed by Granite4VisionNaflexLLMModel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mrope_config: dict | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Granite4VisionNaflexLLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"), mrope_config=mrope_config
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            logit_scale = getattr(config, "logit_scale", 1.0)
            if hasattr(config, "logits_scaling"):
                logit_scale /= config.logits_scaling
            self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
        else:
            self.lm_head = PPMissingLayer()

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        tensors = super().make_empty_intermediate_tensors(batch_size, dtype, device)
        for llm_layer in getattr(self, "_ds_layer_indices", []):
            tensors.tensors[f"ds_{llm_layer}"] = torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            )
        return tensors


# ---------------------------------------------------------------------------
# Multimodal processing info / processor
# ---------------------------------------------------------------------------


class Granite4VisionNaflexProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        """Token count for memory-profiling dummy inputs, delegating to the
        real HF processor's own _get_number_of_features (window-rounding,
        aspect-preserving grid search) rather than approximating it -- the
        exact grid search is nontrivial to reproduce and the processor is
        already available via get_hf_processor().

        Each deepstack/spatial level packs to the SAME pooled grid and is
        stacked along the feature dim (not concatenated as extra tokens) in
        embed_input_ids, so the visible placeholder-token count corresponds
        to a single level's pooled grid.
        """
        processor = self.get_hf_processor()
        return processor._get_number_of_features(image_height, image_width)

    def get_image_size_with_most_features(self) -> ImageSize:
        """NaFlex has no single fixed vision-encoder square size (unlike
        LLaVA, whose get_image_size_with_most_features reads a fixed tower
        size) -- the worst case for placeholder-token count is the largest
        SQUARE image the processor will accept before its max_num_patches
        budget forces it to downscale. A square is the appropriate worst
        case here (for a fixed patch budget, get_num_image_tokens is
        maximized as the aspect ratio approaches 1:1 after the window-side
        rounding).

        MUST round side_patches down to a window_side multiple -- the
        downsampler's windowing (_win_rect) requires H and W divisible by
        window_side, and the real processor enforces this via
        _round_to_window_preserve_aspect. This method's job is to describe
        an image the processor could ACTUALLY produce, not just one under
        the raw patch budget.
        """
        image_processor = self.get_hf_processor().image_processor
        patch_size = image_processor.patch_size
        max_num_patches = getattr(image_processor, "max_num_patches", 5760)
        hf_config = self.get_hf_config()
        window_side = int(hf_config.downsample_rate.split("/")[1]) if hf_config.downsample_rate else 1

        side_patches = max(1, int(max_num_patches**0.5))
        side_patches = max(window_side, (side_patches // window_side) * window_side)
        side_px = side_patches * patch_size
        return ImageSize(width=side_px, height=side_px)


class Granite4VisionNaflexDummyInputsBuilder(LlavaDummyInputsBuilder):
    pass


class Granite4VisionNaflexMultiModalProcessor(
    BaseMultiModalProcessor[Granite4VisionNaflexProcessingInfo]
):
    def _get_hf_processor_text(self, mm_counts: Mapping[str, int]) -> str:
        # vLLM's encoder-budget profiling calls the HF processor with
        # images=[...], text=None by default (BaseMultiModalProcessor's own
        # default). The model card's Granite4VisionNaflexProcessor.__call__
        # only tolerates text=None for some transformers versions -- always
        # supply real dummy text, mirroring LlavaNextMultiModalProcessor's
        # own override (found needed via a granite4_vision bug -- see
        # gv4_release/models/gv5.md).
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # The HF processor (Siglip2ImageNativeProcessor.preprocess) pad-free
        # packs ALL images from one __call__ into a single
        # (1, total_patches, patch_dim) tensor -- there is no per-image
        # leading dimension to slice with .batched(). Instead split along
        # dim=1 by each image's own patch count (H_i * W_i from
        # spatial_shapes), mirroring how Qwen2VL's
        # _create_qwen2vl_field_factory uses flat_from_sizes for its own
        # packed (total_patches, patch_dim) pixel_values (there sliced on
        # dim=0 because Qwen2VL's processor has no leading dummy batch dim;
        # this processor's does, hence dim=1 here). spatial_shapes itself is
        # genuinely one row per image -> .batched().
        spatial_shapes = hf_inputs.get("spatial_shapes", torch.empty((0, 2), dtype=torch.long))
        patch_counts = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", patch_counts, dim=1),
            spatial_shapes=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ):
        """Tells vLLM's framework where the image placeholder tokens land in
        the prompt and how many there are per image -- used both to
        validate/replace <image> when the HF processor is bypassed (cached
        path) and to build PlaceholderRange bookkeeping for every path. The
        HF processor computes the SAME count via _get_number_of_features;
        get_num_image_tokens (on ProcessingInfo) mirrors it."""
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", (ImageProcessorItems,))
            image_size = images.get_image_size(item_idx)
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width, image_height=image_size.height
            )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Granite4VisionNaflexMultiModalProcessor,
    info=Granite4VisionNaflexProcessingInfo,
    dummy_inputs=Granite4VisionNaflexDummyInputsBuilder,
)
class Granite4VisionNaflexForConditionalGeneration(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP, SupportsMRoPE
):
    """vLLM implementation of Granite Vision 5 (granite4_vision_naflex).

    Architecture:
    - SigLIP2 NaFlex vision tower (packed variable-res patches) ->
      NaflexWindowQFormer projectors (one per deepstack level + one per
      spatial-offset group), each owning its own learnable newline /
      optional 2D position embedding.
    - DeepStack: N vision layers projected and injected at N LLM layers;
      spatial: 4 offset groups from the last vision layer injected at 4
      more LLM layers.
    - Granite language backbone with embedding_multiplier, optional 3-axis
      image M-RoPE.
    - logits_scaling via LogitsProcessor.

    Requires `hf_overrides=granite4_vision_naflex.hf_overrides` when the
    checkpoint's text_config has no `mrope_section` in rope_parameters --
    see this module's hf_overrides() docstring.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.layerwise_projectors.": "layerwise_projectors.",
            "model.spatial_projectors.": "spatial_projectors.",
            "model.vision_tower.": "vision_tower.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError(f"Only image modality is supported, got {modality}")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["layerwise_projectors", "spatial_projectors"],
            tower_model="vision_tower",
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = Siglip2NaflexVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

            cache_config = vllm_config.cache_config

            self.layerwise_projectors = nn.ModuleList(
                [
                    NaflexWindowQFormerDownsampler(
                        config,
                        quant_config=quant_config,
                        cache_config=cache_config,
                        prefix=maybe_prefix(prefix, f"layerwise_projectors.{i}"),
                    )
                    for i in range(len(config.deepstack_layer_map))
                ]
            )

            self.spatial_projectors = None
            spatial_target_layers = getattr(config, "spatial_target_layers", [])
            if spatial_target_layers:
                self.spatial_projectors = nn.ModuleList(
                    [
                        NaflexWindowQFormerDownsampler(
                            config,
                            quant_config=quant_config,
                            cache_config=cache_config,
                            spatial_offset=i,
                            prefix=maybe_prefix(prefix, f"spatial_projectors.{i}"),
                        )
                        for i in range(4)
                    ]
                )

        mrope_config = None
        if getattr(config, "use_image_mrope", False):
            mrope_config = {
                "axes": getattr(config, "image_mrope_axes", 2),
                "t_channels": getattr(config, "image_mrope_t_channels", None),
                "method": getattr(config, "image_mrope_method", "round_robin"),
            }

        with self._mark_language_model(vllm_config):
            self.language_model = Granite4VisionNaflexLLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
                mrope_config=mrope_config,
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        self._deepstack_layer_map = config.deepstack_layer_map
        self._spatial_vision_layer = getattr(config, "spatial_vision_layer", -1)
        self._spatial_target_layers = spatial_target_layers
        self._downsample_rate = Fraction(config.downsample_rate)

        self._ds_layer_indices: list[int] = [
            llm_layer for _, llm_layer in config.deepstack_layer_map
        ] + list(spatial_target_layers)
        self.language_model._ds_layer_indices = self._ds_layer_indices

        n_layerwise = len(config.deepstack_layer_map)
        n_spatial = len(spatial_target_layers)
        num_ds_levels = n_layerwise + n_spatial
        lm_hidden = config.text_config.hidden_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._ds_buffers: list[torch.Tensor] = [
            torch.zeros(max_tokens, lm_hidden) for _ in range(num_ds_levels)
        ]
        self._ds_num_tokens: int = 0

        self._use_image_mrope = getattr(config, "use_image_mrope", False)
        self._image_mrope_axes = getattr(config, "image_mrope_axes", 2)
        self._image_mrope_coords = getattr(config, "image_mrope_coords", "corner")
        self._image_token_id = config.image_token_index

    # -----------------------------------------------------------------
    # Vision + downsample -> per-image, per-level packed features
    # -----------------------------------------------------------------

    def _split_per_image(self, features: torch.Tensor, spatial_shapes: torch.Tensor) -> list[torch.Tensor]:
        patch_counts = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        return list(torch.split(features.squeeze(0), patch_counts, dim=0))

    def _get_all_layer_features(
        self, pixel_values: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> tuple[list[int], list[torch.Tensor]]:
        all_hidden_states = self.vision_tower(
            pixel_values, spatial_shapes, output_hidden_states=True
        )

        levels: list[tuple[int, list[torch.Tensor]]] = []
        for proj_idx, (vision_layer, llm_layer) in enumerate(self._deepstack_layer_map):
            selected = all_hidden_states[vision_layer]
            per_image = self._split_per_image(selected, spatial_shapes)
            packed = self.layerwise_projectors[proj_idx](per_image, spatial_shapes)
            levels.append((llm_layer, packed))

        if self.spatial_projectors is not None:
            spatial_hidden = all_hidden_states[self._spatial_vision_layer]
            spatial_per_image = self._split_per_image(spatial_hidden, spatial_shapes)
            for group_idx, llm_layer in enumerate(self._spatial_target_layers):
                packed = self.spatial_projectors[group_idx](spatial_per_image, spatial_shapes)
                levels.append((llm_layer, packed))

        llm_layer_indices = [llm_layer for llm_layer, _ in levels]
        num_images = len(spatial_shapes)
        per_image_packed = [
            torch.cat([levels[lvl][1][img] for lvl in range(len(levels))], dim=-1)
            for img in range(num_images)
        ]
        return llm_layer_indices, per_image_packed

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "spatial_shapes": spatial_shapes}

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        pixel_values = image_input["pixel_values"]
        spatial_shapes = image_input["spatial_shapes"]
        # pixel_values items come from MultiModalFieldConfig.flat_from_sizes(dim=1) --
        # each item is a (1, patches_i, patch_dim) slice of the HF processor's
        # pad-free-packed (1, total_patches, patch_dim) tensor (leading dim kept:
        # flat_from_sizes's dim=1 slice is `data[:, start:end]` on a 3D tensor,
        # not a dim-0 slice). Re-pack by concatenating along dim=1, restoring
        # the single (1, total_patches, patch_dim) sequence the vision tower
        # expects across ALL images in the request.
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=1)
        elif pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)
        # spatial_shapes comes from MultiModalFieldConfig.batched("image"):
        # each per-item element is one (2,) row (H, W), not (1, 2) --
        # reduce_data stacks same-shape items into a single (num_images, 2)
        # tensor when it can, but falls back to a plain list of (2,) rows
        # when it can't. MUST use torch.stack here, not torch.cat(dim=0) --
        # cat on a list of (2,) rows silently flattens to (num_images*2,),
        # matching Qwen2VL's image_grid_thw handling in qwen2_vl.py's
        # _parse_and_validate_image_input.
        if isinstance(spatial_shapes, list):
            spatial_shapes = torch.stack(spatial_shapes)

        llm_layer_indices, per_image_packed = self._get_all_layer_features(
            pixel_values, spatial_shapes
        )
        self._ds_layer_indices = llm_layer_indices
        return per_image_packed

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        lm_inner = self.language_model.model

        has_vision = (
            multimodal_embeddings is not None
            and is_multimodal is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal.any()
        )

        if not has_vision:
            self._ds_num_tokens = 0
            embeds = lm_inner.embed_input_ids(input_ids)
            return embeds * lm_inner.config.embedding_multiplier

        text_embeds = lm_inner.embed_input_ids(input_ids)
        text_embeds[is_multimodal] = 0.0
        inputs_embeds = text_embeds * lm_inner.config.embedding_multiplier

        N, lm_h = inputs_embeds.shape
        all_packed = torch.cat(
            [t.to(dtype=inputs_embeds.dtype) for t in multimodal_embeddings], dim=0
        )
        level_features = all_packed.split(lm_h, dim=-1)

        buf0 = self._ds_buffers[0]
        if buf0.device != inputs_embeds.device or buf0.dtype != inputs_embeds.dtype:
            self._ds_buffers = [
                b.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                for b in self._ds_buffers
            ]

        for level_idx in range(len(self._ds_layer_indices)):
            target = self._ds_buffers[level_idx][:N]
            target.zero_()
            target[is_multimodal] = level_features[level_idx]

        self._ds_num_tokens = N
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if inputs_embeds is not None and get_pp_group().is_first_rank and self._ds_layer_indices:
            n = inputs_embeds.size(0)
            ds: IntermediateTensors | None = IntermediateTensors(
                {
                    f"ds_{llm_layer}": self._ds_buffers[lvl][:n]
                    for lvl, llm_layer in enumerate(self._ds_layer_indices)
                }
            )
        else:
            ds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=ds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank and self._ds_num_tokens > 0:
            n = self._ds_num_tokens
            for buf in self._ds_buffers:
                buf[:n].zero_()
            self._ds_num_tokens = 0

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # -----------------------------------------------------------------
    # SupportsMRoPE -- image M-RoPE position construction
    #
    # Ports the model card's mrope.py:_expand_image_coords +
    # build_mrope_position_ids for the SINGLE-VIEW NaFlex case (one
    # ViewBlock per image -- no base-thumbnail/hires-tile split, unlike the
    # anyres multi-view scheme in image_view_blocks.py, which this
    # architecture's forward() does not use).
    # -----------------------------------------------------------------

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        if not self._use_image_mrope:
            positions = torch.arange(len(input_tokens)).unsqueeze(0).expand(3, -1).clone()
            return positions, 0

        kwargs = MultiModalFeatureSpec.gather_kwargs(mm_features, {"spatial_shapes"})
        spatial_shapes_list = [
            item.tolist() if torch.is_tensor(item) else list(item)
            for item in kwargs.get("spatial_shapes", [])
        ]

        ds = float(self._downsample_rate)
        has_nl = getattr(self.config, "use_image_newline", False)
        image_blocks = [
            [_ViewBlock("base", int(H * ds), int(W * ds), has_nl)] for H, W in spatial_shapes_list
        ]

        input_ids = torch.tensor(input_tokens).unsqueeze(0)
        planes = _build_mrope_position_ids(
            input_ids,
            self._image_token_id,
            image_blocks,
            coords=self._image_mrope_coords,
            n_axes=3,
        )  # [3, 1, S]
        llm_positions = planes[:, 0]  # [3, S]

        h_plane = llm_positions[mrope_scalar_plane_index(3)]
        mrope_position_delta = int(h_plane.max().item()) + 1 - len(input_tokens)
        return llm_positions, mrope_position_delta


# ---------------------------------------------------------------------------
# M-RoPE position-id construction helpers (ported from the model card's
# mrope.py / image_view_blocks.py, single-view NaFlex case only -- see class
# docstring above)
# ---------------------------------------------------------------------------


class _ViewBlock(NamedTuple):
    kind: str
    grid_h: int
    grid_w: int
    has_newline: bool


def _expand_image_coords(blocks: list[_ViewBlock], coords: str = "corner"):
    if coords not in ("corner", "band_center"):
        raise ValueError(f"unknown mrope coords layout: {coords!r}")
    parts = []
    running = 0
    for b in blocks:
        gh, gw = b.grid_h, b.grid_w
        if gh == 0 or gw == 0:
            continue
        cols_per_row = gw + 1 if b.has_newline else gw
        rows = torch.arange(gh).repeat_interleave(cols_per_row)
        cols = torch.arange(cols_per_row).repeat(gh)
        if coords == "band_center":
            m = max(gh, cols_per_row)
            rows = rows + (m - gh) / 2
            cols = cols + (m - cols_per_row) / 2
        parts.append(torch.stack([running + rows, running + cols], dim=-1))
        running = running + max(gh - 1, cols_per_row - 1) + 1
    if not parts:
        return torch.zeros(0, 2, dtype=torch.long), running
    out = torch.cat(parts, dim=0)
    return (out if coords == "band_center" else out.to(torch.long)), running


def _build_mrope_position_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_blocks_per_image: list[list[_ViewBlock]],
    coords: str = "corner",
    n_axes: int = 3,
) -> torch.Tensor:
    assert input_ids.shape[0] == 1, "expects pad-free packed input (B==1)"
    if n_axes not in (2, 3):
        raise ValueError(f"unsupported mrope n_axes: {n_axes!r} (expected 2 or 3)")
    device = input_ids.device
    S = input_ids.shape[1]
    ids = input_ids[0]
    is_img = ids == image_token_id

    coord_dtype = torch.float32 if (coords == "band_center" or n_axes == 3) else torch.long
    row = torch.zeros(S, dtype=coord_dtype, device=device)
    col = torch.zeros(S, dtype=coord_dtype, device=device)
    tvec = torch.zeros(S, dtype=coord_dtype, device=device)
    img_pos = is_img.nonzero(as_tuple=True)[0]

    advance = torch.ones(S, dtype=torch.long, device=device)
    advance[is_img] = 0

    if img_pos.numel() > 0:
        per_image = [_expand_image_coords(b, coords=coords) for b in image_blocks_per_image]
        all_coords = torch.cat([c for c, _ in per_image], dim=0).to(device=device, dtype=coord_dtype)
        assert all_coords.shape[0] == img_pos.numel(), (
            f"coord/token mismatch: {all_coords.shape[0]} coords vs {img_pos.numel()} image tokens"
        )
        row[img_pos] = all_coords[:, 0]
        col[img_pos] = all_coords[:, 1]

        off = 0
        for c, extent in per_image:
            n = c.shape[0]
            last_idx = int(img_pos[off + n - 1].item())
            advance[last_idx] = extent
            if n_axes == 3:
                tvec[img_pos[off : off + n]] = (extent - 1) / 2
            off += n

    boundary = torch.zeros(S, dtype=torch.bool, device=device)
    boundary[0] = True
    adv_before = torch.zeros(S, dtype=torch.long, device=device)
    adv_before[1:] = advance[:-1]
    adv_before = adv_before.masked_fill(boundary, 0)
    csum = torch.cumsum(adv_before, dim=0)
    seg_start = torch.where(boundary, csum, torch.zeros_like(csum))
    base = csum - torch.cummax(seg_start, dim=0).values

    base = base.to(coord_dtype)
    h = torch.where(is_img, base + row, base)
    w = torch.where(is_img, base + col, base)
    if n_axes == 2:
        return torch.stack([h, w]).unsqueeze(1)
    t = base + tvec
    return torch.stack([t, h, w]).unsqueeze(1)
