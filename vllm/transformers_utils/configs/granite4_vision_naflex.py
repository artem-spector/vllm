# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import transformers


class Granite4VisionNaflexConfig(transformers.PretrainedConfig):
    """Configuration for Granite Vision 5 (granite4_vision_naflex).

    This config is needed because the granite4_vision_naflex model type is not
    registered anywhere in transformers (unlike granite4_vision, which is a
    native transformers model type as of v5.8.0). Once transformers adds
    native support, this file can be removed and the _CONFIG_REGISTRY entry
    dropped, mirroring granite4_vision's own history.
    """

    model_type = "granite4_vision_naflex"
    is_composition = False

    def __init__(
        self,
        vision_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        qformer_config: dict[str, Any] | None = None,
        image_token_index: int = 100352,
        image_seq_length: int = 576,
        image_grid_pinpoints: list[list[int]] | None = None,
        vision_feature_select_strategy: str = "full",
        vision_feature_layer: int | list[int] = -2,
        projector_dropout: float = 0.1,
        downsample_rate: str | None = None,
        use_image_newline: bool = False,
        use_projector_pos_embed: bool = False,
        projector_pos_embed_grid_size: int = 32,
        deepstack_layer_map: list[list[int]] | None = None,
        spatial_vision_layer: int = -1,
        spatial_target_layers: list[int] | None = None,
        use_image_mrope: bool = False,
        image_mrope_axes: int = 2,
        image_mrope_coords: str = "corner",
        image_mrope_method: str = "round_robin",
        image_mrope_t_channels: int | None = None,
        pretrained_vision_tower: str = "",
        pretrained_language_model: str = "",
        **kwargs: Any,
    ):
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.image_grid_pinpoints = image_grid_pinpoints or []
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.projector_dropout = projector_dropout
        self.downsample_rate = downsample_rate
        self.use_image_newline = use_image_newline
        self.use_projector_pos_embed = use_projector_pos_embed
        self.projector_pos_embed_grid_size = projector_pos_embed_grid_size
        self.deepstack_layer_map = deepstack_layer_map or []
        self.spatial_vision_layer = spatial_vision_layer
        self.spatial_target_layers = spatial_target_layers or []
        self.use_image_mrope = use_image_mrope
        self.image_mrope_axes = image_mrope_axes
        self.image_mrope_coords = image_mrope_coords
        self.image_mrope_method = image_mrope_method
        self.image_mrope_t_channels = image_mrope_t_channels
        self.pretrained_vision_tower = pretrained_vision_tower
        self.pretrained_language_model = pretrained_language_model

        vision_config = dict(vision_config or {})
        vision_model_type = vision_config.get("model_type", "siglip2_vision_model")
        if vision_model_type in transformers.CONFIG_MAPPING:
            self.vision_config = transformers.CONFIG_MAPPING[vision_model_type](
                **vision_config
            )
        else:
            self.vision_config = transformers.PretrainedConfig(**vision_config)

        text_config = dict(text_config or {})
        text_model_type = text_config.get("model_type", "granite")
        if text_model_type in transformers.CONFIG_MAPPING:
            self.text_config = transformers.CONFIG_MAPPING[text_model_type](
                **text_config
            )
        else:
            self.text_config = transformers.PretrainedConfig(**text_config)

        # blip_2_qformer is in transformers' CONFIG_MAPPING (used for the
        # window-qformer downsamplers) -- no special-casing needed like
        # vision/text above.
        qformer_config = dict(qformer_config or {})
        qformer_model_type = qformer_config.get("model_type", "blip_2_qformer")
        if qformer_model_type in transformers.CONFIG_MAPPING:
            self.qformer_config = transformers.CONFIG_MAPPING[qformer_model_type](
                **qformer_config
            )
        else:
            self.qformer_config = transformers.PretrainedConfig(**qformer_config)

        super().__init__(**kwargs)
