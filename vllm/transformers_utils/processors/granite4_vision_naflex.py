# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor + image processor for Granite Vision 5 (granite4_vision_naflex).

Needed because granite4_vision_naflex is not registered anywhere in
transformers (unlike granite4_vision's LlavaNext-based processor, which
subclasses a standard, already-registered transformers processor). Once
transformers adds native support, this file can be removed, the
_CLASS_TO_MODULE entries in __init__.py dropped, and the AutoImageProcessor
registration below deleted.

vLLM's own get_image_processor() (vllm/transformers_utils/processor.py) has
no registry bypass of its own -- it calls AutoImageProcessor.from_pretrained
directly. So the image processor side is registered via transformers' own
extension point, AutoImageProcessor.register(), called at import time below
(mirrors vllm/transformers_utils/configs/ovis.py's module-level
AutoConfig.register() pattern for the analogous config-side gap).
"""

import math
from fractions import Fraction

import numpy as np
import torch
import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_backends import np_normalize, np_rescale
from transformers.image_transforms import (
    convert_to_rgb,
    resize as _resize_fn,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
)
from transformers.models.siglip2 import Siglip2ImageProcessor
from transformers.models.siglip2.image_processing_siglip2 import (
    convert_image_to_patches,
)
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack

from vllm.transformers_utils.configs.granite4_vision_naflex import (
    Granite4VisionNaflexConfig,
)

__all__ = ["Granite4VisionNaflexProcessor"]

_OFFSETS_DOWNSCALED = (-2, -1, 0, 1)
_OFFSETS_UPSCALED = (-1, 0, 1, 2)


def _round_to_window_preserve_aspect(
    target_h, target_w, window_side, max_num_patches=None, min_num_patches=None, scale=1.0
):
    log_aspect = math.log(target_w / target_h)
    target_count = target_h * target_w
    h_windows = round(target_h / window_side)
    w_windows = round(target_w / window_side)
    offsets = _OFFSETS_DOWNSCALED if scale < 1.0 else _OFFSETS_UPSCALED

    best, best_err = None, None
    for dh in offsets:
        for dw in offsets:
            nh, nw = h_windows + dh, w_windows + dw
            if nh < 1 or nw < 1:
                continue
            h, w = nh * window_side, nw * window_side
            count = h * w
            over = max_num_patches is not None and count > max_num_patches
            under = min_num_patches is not None and count < min_num_patches
            out_of_budget = 1 if (over or under) else 0
            err = (
                out_of_budget,
                abs(math.log(w / h) - log_aspect),
                abs(math.log(count / target_count)),
            )
            if best_err is None or err < best_err:
                best_err = err
                best = (h, w)
    return best


def _compute_target_patches(
    orig_height, orig_width, patch_size, window_side, max_num_patches, min_num_patches
):
    target_h = orig_height / patch_size
    target_w = orig_width / patch_size

    num_patches = target_h * target_w
    scale = 1.0
    if num_patches < min_num_patches:
        scale = math.sqrt(min_num_patches / num_patches)
    elif num_patches > max_num_patches:
        scale = math.sqrt(max_num_patches / num_patches)
    target_h *= scale
    target_w *= scale

    return _round_to_window_preserve_aspect(
        target_h, target_w, window_side, max_num_patches, min_num_patches, scale=scale
    )


class Granite4VisionNaflexProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False},
        "images_kwargs": {"do_pad": True},
    }


class Siglip2ImageNativeProcessor(Siglip2ImageProcessor):
    """Overrides preprocess() wholesale (not the newer _preprocess() hook),
    for compatibility across transformers versions that may or may not have
    the _preprocess() split (the model card's own processor relies on it
    being present; this fork copy does not need to assume that).

    Mirrors the model card's Siglip2ImageNativeProcessor's numpy pipeline
    (to_numpy_array -> to_channel_dimension_format -> resize -> rescale ->
    normalize) with two differences: window-aware _compute_target_patches
    sizing (so grids stay divisible by the downsampler's window_side)
    instead of the base class's plain max_num_patches sizing, and pad-free
    packing (concatenate patches across images into one
    (1, total_patches, patch_dim) sequence) instead of padding each image to
    a fixed max_num_patches.
    """

    @property
    def window_side(self) -> int:
        dr = getattr(self, "downsample_rate", None)
        return int(dr.split("/")[1]) if dr else 1

    def preprocess(
        self,
        images,
        do_resize=None,
        resample=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        return_tensors=None,
        input_data_format=None,
        do_convert_rgb=None,
        patch_size=None,
        max_num_patches=None,
        **kwargs,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_num_patches = (
            max_num_patches if max_num_patches is not None else self.max_num_patches
        )
        min_num_patches = getattr(self, "min_num_patches", 64)
        window_side = getattr(self, "window_side", 1)

        images = make_flat_list_of_images(images)
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]
        input_data_format = input_data_format or infer_channel_dimension_format(
            images[0]
        )

        all_patches = []
        spatial_shapes = []

        for image in images:
            image = to_channel_dimension_format(
                image, input_data_format, input_channel_dim=input_data_format
            )

            if do_resize:
                target_h, target_w = _compute_target_patches(
                    image.shape[0],
                    image.shape[1],
                    patch_size=patch_size,
                    window_side=window_side,
                    max_num_patches=max_num_patches,
                    min_num_patches=min_num_patches,
                )
                height = target_h * patch_size
                width = target_w * patch_size
                image = _resize_fn(
                    image=image,
                    size=(height, width),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            # Use the module-level np_rescale/np_normalize directly, not
            # self.rescale/self.normalize -- Siglip2ImageProcessor's MRO
            # puts TorchvisionBackend before BaseImageProcessor, so
            # self.rescale/self.normalize resolve to the torch-tensor-only
            # fast-backend versions (crash: "F.normalize supports ... but
            # got numpy.ndarray"). This preprocess() override works in
            # numpy throughout, so it needs the numpy-native functions
            # explicitly.
            if do_rescale:
                image = np_rescale(
                    image=image, scale=rescale_factor, input_data_format=input_data_format
                )
            if do_normalize:
                image = np_normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            spatial_shapes.append((image.shape[0] // patch_size, image.shape[1] // patch_size))
            # convert_image_to_patches expects a channels-FIRST torch.Tensor
            # (num_channels, height, width) -- image is channels-last numpy
            # up to this point (matches how height/width are read as
            # shape[0]/shape[1] just above).
            chw_image = to_channel_dimension_format(
                image, "channels_first", input_channel_dim=input_data_format
            )
            patches = convert_image_to_patches(torch.as_tensor(chw_image), patch_size)
            all_patches.append(patches)

        pixel_values = torch.cat(all_patches, dim=0).unsqueeze(0)
        spatial_shapes_t = torch.tensor(spatial_shapes)

        return BatchFeature(
            data={"pixel_values": pixel_values, "spatial_shapes": spatial_shapes_t},
            tensor_type=return_tensors,
        )


class Granite4VisionNaflexProcessor(ProcessorMixin):
    # ProcessorMixin.attributes defaults to ['feature_extractor', 'tokenizer']
    # -- must override to ['image_processor', 'tokenizer'] or
    # _get_arguments_from_pretrained looks up the unset feature_extractor_class
    # instead of image_processor_class.
    attributes = ["image_processor", "tokenizer"]
    model_type = "granite4_vision_naflex"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        downsample_rate=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.downsample_rate = downsample_rate

    def __call__(
        self,
        images=None,
        text=None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Granite4VisionNaflexProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        output_kwargs = self._merge_kwargs(
            Granite4VisionNaflexProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if image_inputs and text is not None:
            if not isinstance(images[0], list):
                images = [images]
            prompt_strings = []
            for sample, sample_images in zip(text, images):
                image_iter = iter(sample_images)
                while self.image_token in sample:
                    image = next(image_iter)
                    im_shape = to_numpy_array(image).shape
                    num_image_tokens = self._get_number_of_features(im_shape[0], im_shape[1])
                    sample = sample.replace(self.image_token, "<placeholder>" * num_image_tokens, 1)
                prompt_strings.append(sample)
            prompt_strings = [s.replace("<placeholder>", self.image_token) for s in prompt_strings]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", None
        )

        if prompt_strings is None:
            return BatchFeature(data=image_inputs, tensor_type=return_tensors)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_number_of_features(
        self, orig_height: int, orig_width: int, height_=None, width_=None
    ) -> int:
        sig_h, sig_w = _compute_target_patches(
            orig_height,
            orig_width,
            patch_size=self.image_processor.patch_size,
            window_side=getattr(self.image_processor, "window_side", 1),
            max_num_patches=self.image_processor.max_num_patches,
            min_num_patches=getattr(self.image_processor, "min_num_patches", 64),
        )
        ds = getattr(self.image_processor, "downsample_rate", None)
        ratio = Fraction(ds) if ds is not None else Fraction(1)
        pooled_h = int(sig_h * ratio)
        pooled_w = int(sig_w * ratio)
        if getattr(self.image_processor, "use_image_newline", False):
            return pooled_h * pooled_w + pooled_h
        return pooled_h * pooled_w


# Module-level registration -- executes once, the first time this module is
# imported (triggered by vllm.transformers_utils.processors.__getattr__ via
# get_processor()'s `getattr(processors, "Granite4VisionNaflexProcessor")`
# lookup, itself driven by the checkpoint's processor_config.json declaring
# "processor_class": "Granite4VisionNaflexProcessor"). Mirrors
# vllm/transformers_utils/configs/ovis.py's equivalent module-level
# AutoConfig.register() calls.
#
# slow_image_processor_class only -- Siglip2ImageNativeProcessor subclasses
# Siglip2ImageProcessor (numpy/PIL-based, not BaseImageProcessorFast).
# AutoImageProcessor.register() raises ValueError if a non-fast class is
# passed as fast_image_processor_class.
transformers.AutoImageProcessor.register(
    Granite4VisionNaflexConfig,
    slow_image_processor_class=Siglip2ImageNativeProcessor,
    exist_ok=True,
)
