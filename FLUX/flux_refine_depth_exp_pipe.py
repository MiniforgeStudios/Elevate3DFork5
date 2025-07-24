# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import inspect
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from torchao.quantization import quantize_, int8_weight_only
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tqdm import tqdm

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers import FluxPriorReduxPipeline

from FLUX.utils.fft import (extract_low_frequency_and_mask,
                            extract_high_frequency,
                            combine_latents,
                            nc_get_low_or_high_fft,
                            compute_spectrum_pil,
                            compute_rapsd_pil,)

from FLUX.scheduler import FlowMatchEulerDiscreteInvertScheduler

from FLUX.flux_frdiff_mixin import FRDiffMixin

from Utils.etc import RefineType

from Configs import MyFluxPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        >>> source = load_image(img_url)
        >>> mask = load_image(mask_url)
        >>> image = pipe(prompt=prompt, image=source, mask_image=mask).images[0]
        >>> image.save("flux_inpainting.png")
        ```
        ```
        """


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxInpaintPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FRDiffMixin):
    r"""
    The Flux pipeline for image inpainting.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    #model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    model_cpu_offload_seq = "text_encoder->text_encoder_2"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", "x0_pred"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        
        # WARN: MOVE T5 to CPU
        #self.text_encoder_2.to(device)
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        #self.text_encoder_2.to("cpu")

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, timesteps, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def check_inputs(
        self,
        prompt,
        prompt_2,
        strength,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids_ori(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    
        # Assign base row and column IDs
        latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]
    
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    
        # Initialize a tensor to hold all batch IDs
        all_latent_image_ids = torch.zeros(batch_size, latent_image_id_height, latent_image_id_width, latent_image_id_channels, device=device, dtype=dtype)
    
        # Assign unique IDs for each batch
        for batch_idx in range(batch_size):
            batch_offset = batch_idx * max(height // 2, width // 2)
            all_latent_image_ids[batch_idx, ..., 1] = latent_image_ids[..., 1] + batch_offset
            all_latent_image_ids[batch_idx, ..., 2] = latent_image_ids[..., 2] + batch_offset
    
        # Reshape to the desired shape
        all_latent_image_ids = all_latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
    
        return all_latent_image_ids


    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=None,
        image_latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        ori_height, ori_width = height, width

        height = 2 * (int(height) // self.vae_scale_factor)
        width  = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
        # return latents.to(device=device, dtype=dtype), latent_image_ids

        # if image_latents is None:
        #     if latents is None:
        #         image = image.to(device=device, dtype=dtype)
        #         image_latents = self._encode_vae_image(image=image, generator=generator) # NOTE: This depends on a seed... hmm
        #     else:
        #         image_latents = self._unpack_latents(latents, ori_height, ori_width, self.vae_scale_factor)


        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if latents is None:
            # latents = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timestep, noise)
            latents = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype) if is_strength_max else self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            # latents = noise if is_strength_max else self.scheduler.scale_noise(latents, timestep, noise)
            latents = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype) if is_strength_max else self.scheduler.scale_noise(latents, timestep, noise)
        
        noise = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        # WARN: IMPORTANT! MULTI REF IMAGE!
        image_latents = self._pack_latents(image_latents, len(image_latents), num_channels_latents, height, width)
        latents       = self._pack_latents(latents, len(image_latents), num_channels_latents, height, width)
        # NOTE: This is a hack. but since we swap around the reference image, it wont be that harmful.
        latents = latents[[0]]

        return latents, noise, image_latents, latent_image_ids

    # Copied from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet.StableDiffusion3ControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents_by_freq_swap(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=None,
        image_latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        ori_height, ori_width = height, width

        height = 2 * (int(height) // self.vae_scale_factor)
        width  = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
        # return latents.to(device=device, dtype=dtype), latent_image_ids

        if image_latents is None:
            if latents is None:
                image = image.to(device=device, dtype=dtype)
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                #image_latents = latents
                image_latents = self._unpack_latents(latents, ori_height, ori_width, self.vae_scale_factor)
        
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timestep, noise) # PIPOUT latent is used

        noise = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents, noise, image_latents, latent_image_ids

    def prepare_latents_by_inversion(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timesteps=None,
        text_ids=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        is_strength_max=None,
        output_subdir=None,
        gamma = 0.5,
        vanilla = True,
        debug = False,
        inpaint_width = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        ori_height, ori_width = height, width
        ori_inpaint_width = int((inpaint_width / 2) * self.vae_scale_factor)

        height = 2 * (int(height) // self.vae_scale_factor)
        width  = 2 * (int(width) // self.vae_scale_factor)


        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
        ref_latent_image_ids = self._prepare_latent_image_ids(batch_size, height, inpaint_width, device, dtype)

        noise         = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if latents is None:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            forward_latents = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timesteps[0].unsqueeze(0), noise) # PIPOUT latent is used
        else:
            image_latents = self._unpack_latents(latents, ori_height, ori_width, self.vae_scale_factor)
        
        noise         = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        ref_latents   = image_latents[:, :, :, :inpaint_width]
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        ref_latents   = self._pack_latents(ref_latents, batch_size, num_channels_latents, height, inpaint_width)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            # WARN:
            # guidance = torch.tensor([self.guidance_scale], device=device)
            # guidance = guidance.expand(noise.shape[0])
            guidance = torch.tensor([0.], device=device)
            guidance = guidance.expand(noise.shape[0])
        else:
            guidance = None
        
        replace_latents = []

        zt = image_latents.clone().to(self.device)
        #replace_latents.append(zt)
        if not vanilla:
            pbar = tqdm(reversed(timesteps)[:-1], desc='RF inversion')
            for idx, t in enumerate(pbar):
                timestep = t.expand(zt.shape[0]).to(zt.dtype)

                sigma_next   = reversed(self.scheduler.sigmas)[idx + 1]
                sigma        = reversed(self.scheduler.sigmas)[idx]

                noise_pred = self.transformer(
                    hidden_states=zt,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_w_cond  = (noise - zt) / (1. - (timestep / 1000.))
                noise_pred_control = noise_pred + gamma * (noise_pred_w_cond - noise_pred)

                print(f"timestep: {t}, sigma: {sigma}, sigma_next: {sigma_next}")

                zt = zt + noise_pred_control * (sigma_next - sigma)
                replace_latents.append(zt)

                # x0_latents = self._unpack_latents(zt_prev_pred, ori_height, ori_width, self.vae_scale_factor)
                # x0_latents = (x0_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                # image = pipe.vae.decode(x0_latents, return_dict=False)[0]
                # image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                # image.save(output_subdir / f"RF_invert_idx{idx}_timestep_{t}_x0_pred.png")
                
                if debug:
                    latents = self._unpack_latents(zt, ori_height, ori_width, self.vae_scale_factor)
                    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    image = pipe.vae.decode(latents, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                    image.save(output_subdir / f"RF_invert_idx{idx}_timestep_{t}.png")
            #replace_latents.insert(0, zt)
        
        else:
            pbar = tqdm(reversed(timesteps), desc='RF inversion')
            zt = ref_latents.clone().to(self.device)
            for idx, t in enumerate(pbar):
                timestep = t.expand(zt.shape[0]).to(zt.dtype)

                sigma_next   = reversed(self.scheduler.sigmas)[idx + 1]
                sigma        = reversed(self.scheduler.sigmas)[idx]

                noise_pred = self.transformer(
                    hidden_states=zt,
                    timestep=timestep / 1000, #sigma.expand(zt.shape[0]).to(zt.dtype), #timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=ref_latent_image_ids,
                    #img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                print(f"timestep: {t}, 1-t_i: {1. - (timestep / 1000.)}, sigma: {sigma}, sigma_next: {sigma_next}")

                zt = zt + noise_pred * (sigma_next - sigma)
                zt_unpk = self._unpack_latents(zt, ori_height, ori_inpaint_width, self.vae_scale_factor)
                #replace_latents.append(zt_unpk)
                replace_latents.insert(0, zt_unpk)

                if debug:
                    #latents = self._unpack_latents(zt, ori_height, ori_inpaint_width, self.vae_scale_factor)
                    latents = (zt_unpk / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    image = pipe.vae.decode(latents, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                    image.save(output_subdir / f"RF_invert_idx{idx}_timestep_{t}.png")

            #zt = torch.tile(self._unpack_latents(zt, ori_height, ori_inpaint_width, self.vae_scale_factor), (1, 1, 1, 2))

            zt = self._unpack_latents(zt, ori_height, ori_inpaint_width, self.vae_scale_factor)
            #WARN: copy or not
            zt = torch.cat((zt, forward_latents[..., inpaint_width:]), dim=-1)
            #zt = torch.tile(zt, (1, 1, 1, 2))
            #forward_latents
            zt = self._pack_latents(zt, batch_size, num_channels_latents, height, width)
        latents = zt

        latents = latents.to(dtype)
        image_latents = image_latents.to(dtype)

        return latents, noise, image_latents, latent_image_ids, replace_latents

    def prepare_latents_by_nc_sdedit(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timesteps=None,
        text_ids=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        is_strength_max=None,
        output_subdir=None,
        gamma = 0.5,
        vanilla = True,
        debug = False,
        inpaint_width = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        ori_height, ori_width = height, width
        ori_inpaint_width = int((inpaint_width / 2) * self.vae_scale_factor)

        height = 2 * (int(height) // self.vae_scale_factor)
        width  = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
        ref_latent_image_ids = self._prepare_latent_image_ids(batch_size, height, inpaint_width, device, dtype)

        noise         = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if latents is None:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            forward_latents = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timesteps[0].unsqueeze(0), noise) # PIPOUT latent is used
        else:
            image_latents = self._unpack_latents(latents, ori_height, ori_width, self.vae_scale_factor)
        
        noise         = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        ref_latents   = image_latents[:, :, :, :inpaint_width]
        image_latents_unpacked = image_latents

        forward_latents = self._pack_latents(forward_latents, batch_size, num_channels_latents, height, width)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        ref_latents   = self._pack_latents(ref_latents, batch_size, num_channels_latents, height, inpaint_width)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            # WARN:
            guidance = torch.tensor([self.guidance_scale], device=device)
            guidance = guidance.expand(noise.shape[0])
        else:
            guidance = None
        
        for idx in range(3):
            timestep = timesteps[0].expand(forward_latents.shape[0]).to(forward_latents.dtype)

            noise_pred = self.transformer(
                hidden_states=forward_latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            sigma = self.scheduler.sigmas[len(timesteps)]
            x0_pred = (forward_latents - sigma * noise_pred) / (1. - sigma)

            pred_latents_unpacked = self._unpack_latents(x0_pred, ori_height, ori_width, self.vae_scale_factor)
            pred_noise_unpacked   = self._unpack_latents(noise_pred, ori_height, ori_width, self.vae_scale_factor)
                                    
            high_pred_latents     = nc_get_low_or_high_fft(pred_latents_unpacked.float() , scale=0.5, is_low=True).to(forward_latents.dtype)
            high_image_latents    = nc_get_low_or_high_fft(image_latents_unpacked.float(), scale=0.5, is_low=True).to(forward_latents.dtype)
            
            x0_latents = (pred_latents_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            image = pipe.vae.decode(x0_latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")[0]
            image.save(output_subdir / f"nc-sdedit_x0_idx_{idx}_timestep_{timestep.item()}.png")

            # TODO: Do NC
            # new_noise = pred_noise_unpacked + (high_image_latents - high_pred_latents) * (1. - sigma) / sigma
            # forward_latents = pred_latents_unpacked * (1. - sigma) + sigma * new_noise
            
            forward_latents = (pred_latents_unpacked + (high_pred_latents - high_image_latents)) * (1. - sigma) + sigma * pred_noise_unpacked
            forward_latents = self._pack_latents(forward_latents, batch_size, num_channels_latents, height, width)


        latents = forward_latents
        latents = latents.to(dtype)
        image_latents = image_latents.to(dtype)

        return latents, noise, image_latents, latent_image_ids
    
    def pil_to_latent(self, image, height, width, device, dtype, generator, crops_coords=None, resize_mode="default"):
        image = image.convert('RGB')
        image = self.image_processor.preprocess(
            [image], height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        image = image.to(dtype=torch.float32)

        ori_height, ori_width = height, width

        height = 2 * (int(height) // self.vae_scale_factor)
        width  = 2 * (int(width) // self.vae_scale_factor)

        num_channels_latents = self.transformer.config.in_channels // 4
        batch_size = len(image)
        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator) # NOTE: This depends on a seed... hmm

        return image_latents

    def prepare_mask(
        self,
        mask,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(2 * height // self.vae_scale_factor, 2 * width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(2 * height // self.vae_scale_factor, 2 * width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 16:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

def center_crop(im, new_width, new_height):
    """
    Center crops an image to the specified width and height.

    :param image_path: Path to the input image.
    :param new_width: Desired width of the cropped image.
    :param new_height: Desired height of the cropped image.
    :return: The cropped PIL Image object.
    """
    width, height = im.size

    if new_width > width or new_height > height:
        raise ValueError("New dimensions must be smaller than the original dimensions.")

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_im = im.crop((left, top, right, bottom))

    return cropped_im


def initialize_flux_pipeline(cfg: Dict[str, Any], device: str = 'cuda') -> FluxInpaintPipeline:
    """
    Initializes the FLUX Inpainting Pipeline.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing FLUX settings.
        device (str, optional): Device to load the models onto. Defaults to 'cuda'.

    Returns:
        FluxInpaintPipeline: The initialized FLUX Inpainting Pipeline.
    """
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.flux.depth_pipeline_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    quantize_(transformer, int8_weight_only())

    text_encoder_2 = T5EncoderModel.from_pretrained(
        cfg.flux.depth_pipeline_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    quantize_(text_encoder_2, int8_weight_only())

    # Initialize the Inpainting Pipeline
    pipe = FluxInpaintPipeline.from_pretrained(
        cfg.flux.depth_pipeline_path,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
        #use_safetensors=True
    ).to(device)

    # Configure the Scheduler
    pipe.scheduler = FlowMatchEulerDiscreteInvertScheduler.from_config(pipe.scheduler.config)
    
    if cfg.flux.use_im_prompt:
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            cfg.flux.redux_pipeline_path,
            torch_dtype=torch.bfloat16,).to(device)
    else:
        pipe_prior_redux=None

    # pipe = FluxInpaintPipeline.from_pretrained(
    #         cfg.flux.base_pipeline_path,
    #         torch_dtype=torch.bfloat16,
    #         use_safetensors=True,
    #         # text_encoder=None,
    #         # text_encoder_2=None,
    #         ).to('cuda')

    # pipe.scheduler = FlowMatchEulerDiscreteInvertScheduler.from_config(pipe.scheduler.config)
    
    my_pipe = MyFluxPipeline(base_pipe=pipe, redux_pipe=pipe_prior_redux)
    
    return my_pipe

def run_flux_pipeline(
    cfg: Dict[str, Any],
    my_pipe: MyFluxPipeline,
    exp_in_dir: Path,
    input_dir: Path,
    ref_dir: Path,
    output_subdir: Path,
    postfix: str,
    is_init: bool,
    refine_type: RefineType,
    # first_image: Image.Image = None,
    # first_control_image: Image.Image = None,
    np_image: Image.Image = None,
    np_control: Image.Image = None,
) -> Image.Image:
    """
    Configures the FLUX pipeline with new input images and masks.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing FLUX settings.
        pipe (FluxInpaintPipeline): The initialized FLUX Inpainting Pipeline.
        input_dir (Path): Directory containing input images and masks.
        output_subdir (Path): Directory to save processed images and masks.

    Returns:
        Tuple[List[Image.Image], List[Image.Image], int, int]: 
            - List of initial images.
            - List of initial masks.
            - Width of the images.
            - Height of the images.
    """
    pipe = my_pipe.base_pipe
    redux_pipe = my_pipe.redux_pipe

    print(f"flux pipeline: refine type: {refine_type}")
    if refine_type == RefineType.SIGNIFICANT:
        flux_cfg = cfg.flux.significant
    elif refine_type == RefineType.MINOR:
        flux_cfg = cfg.flux.minor
    elif refine_type == RefineType.NEGLIGIBLE:
        flux_cfg = cfg.flux.negligible
    else:
        return

    # Read Prompt
    with open(exp_in_dir / 'prompt.txt', 'r', encoding="utf-8") as f:
        prompt = f.read()
    
    if is_init:
        p_prompt = (
            f"Orthographic camera. High quality professional photograph. Extremely detailed photograph. "
            f"Image with a {cfg.flux.bg_color} background. "
        )

    elif cfg.flux.use_grid:
        p_prompt = (
            f"Orthographic camera. A two-view image grid. "
            f"Image with a {cfg.flux.bg_color} background. "
        )

    if cfg.flux.use_simple_prompt:
        prompt = p_prompt
    else:
        prompt = p_prompt + prompt
    prompt_2 = prompt

    # Load Input Image
    input_image = Image.open(input_dir / f'rgba_{postfix}.png').convert('RGBA')
    fg_mask           = input_image.split()[-1]
    fg_mask_np        = np.array(fg_mask)

    np_fg_mask        = np_image.split()[-1]
    np_fg_mask_np     = np.array(np_fg_mask)

    # Load depth Image
    control_image = Image.open(input_dir / f'depth_{postfix}.png').convert('RGB')
    
    # Load Mask Images
    # mask_images = [
    #     Image.open(m_path).convert("RGBA") 
    #     for m_path in sorted(input_dir.glob('cos_thresh_mask_*.png'))
    # ]

    mask_images = [Image.open(input_dir / f'cos_thresh_mask_{postfix}_thresh_{cfg.cos_thresh}.png').convert('RGB')]
    
    # Load Reference Images or Create White Background
    if is_init:
        #ref_images = [Image.new("RGBA", input_image.size, (255, 255, 255, 255))]
        ref_images = [input_image]
    else:
        ref_images = [
            Image.open(ref_path).convert("RGBA") 
            for ref_path in sorted(ref_dir.glob('*.png'))
        ]
        
    # Define Background Application
    bg_color = cfg.flux.bg_color
    bg_color_options = cfg.flux.bg_color_options

    apply_background = lambda img: Image.alpha_composite(
        Image.new("RGBA", img.size, tuple(bg_color_options[bg_color])),
        img
    )
    
    input_image = apply_background(input_image)
    np_image    = apply_background(np_image.convert('RGBA'))
    ref_images  = [apply_background(ref_image) for ref_image in ref_images]

    if cfg.flux.use_im_prompt:
        pipe_prior_output = redux_pipe(
            [ref_image.convert("RGB") for ref_image in ref_images],
            prompt=[prompt] * len(ref_images), 
            )
    
    # Save Processed Images
    input_image.save(output_subdir / "input.png")
    for i, mask_image in enumerate(mask_images):
        mask_image.save(output_subdir / f'mask_image_{i:03}.png')
    for i, ref_image in enumerate(ref_images):
        ref_image.save(output_subdir / f'ref_image_{i:03}.png')

    #if not is_init and cfg.flux.use_grid:

    if cfg.flux.use_grid:
        control_image = Image.fromarray(
                np.concatenate((np.array(np_control.convert('RGB')), np.array(control_image)), axis=1)
            ).convert("RGB")

    # if is_init or not cfg.flux.use_grid:
    #     init_images = [input_image.convert("RGB")]
    # else:

    if cfg.flux.use_grid:
        init_images = [
            Image.fromarray(
                np.concatenate((np.array(np_image.convert("RGBA")), np.array(input_image)), axis=1)
            ).convert("RGB")
        ]
    else:
        init_images = [input_image.convert("RGB")]

        # init_images = [
        #     Image.fromarray(
        #         np.concatenate((np.array(ref_image), np.array(input_image)), axis=1)
        #     ).convert("RGB")
        #     for ref_image in ref_images
        # ]

    # Create Masks
    black_mask = Image.new('RGB', ref_images[0].size, (0, 0, 0, 255))  # Black with full opacity
    white_mask = Image.new('RGB', input_image.size, (255, 255, 255, 255))  # White with full opacity
    inpaint_width = int(input_image.size[0] / 8)
    
    if is_init:
        # init_masks = [
        #     Image.fromarray(
        #         np.concatenate((np_fg_mask_np, fg_mask_np), axis=1)
        #     ).convert("RGB")
        # ]

        init_masks = [Image.new('RGB', init_images[0].size, (255, 255, 255))]  # White with full opacity
    else:
        if cfg.flux.use_grid:
            init_masks = [
                Image.fromarray(
                    np.concatenate((np.array(black_mask), np.array(mask_image)), axis=1)
                ).convert("RGB")
                for mask_image in mask_images
            ]
        else:
            init_masks = mask_images
    
    if is_init:
        half_masks = None #init_masks
    else:
        half_masks = [
            Image.fromarray(
                np.concatenate((np.array(black_mask), np.array(white_mask)), axis=1)
            ).convert("RGB")
            for mask_image in mask_images
        ]
    
    # Save Concatenated Images and Masks
    for i, img in enumerate(init_images):
        img.save(output_subdir / f'init_image_{i:03}.png')
    for i, init_mask in enumerate(init_masks):
        init_mask.save(output_subdir / f'init_mask_{i:03}.png')
    control_image.save(output_subdir / f'control_image_{i:03}.png')
    # Get Image Dimensions
    width, height = init_images[0].size
    
    # Save Prompt
    (output_subdir / "prompt.txt").write_text(prompt, encoding='utf-8')

    # input_image       = Image.open(output_subdir / 'init_image_000.png').convert("RGBA")
    # control_image     = Image.open(output_subdir / 'control_image_000.png').convert("RGB")
    # apply_background  = lambda img: Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255)), img)
    # input_image       = apply_background(input_image).convert('RGB')
    # m_path            = output_subdir / 'init_mask_000.png'
    # mask = Image.open(m_path).convert("RGBA")

    if cfg.flux.use_im_prompt:
        image = pipe(
            image=init_images,
            control_image=control_image,
            mask_image=init_masks,
            half_mask=half_masks,
            strength=flux_cfg.strength,
            #prompt=prompt,
            **pipe_prior_output,
            height=height,
            width =width,
            num_inference_steps=flux_cfg.steps,
            guidance_scale=flux_cfg.guidance_scale,
            generator=torch.Generator().manual_seed(cfg.seed),
            output_subdir=output_subdir,
            is_grid=(cfg.flux.use_grid and not is_init),
            is_init=is_init,
            debug=False,
            low_freq_ratio=flux_cfg.low_freq_ratio,
            replace_type=flux_cfg.replace_type,
            replace_step=flux_cfg.replace_steps,
            replace_limit=flux_cfg.replace_limit,
            stop_replace_step=flux_cfg.stop_replace_steps,
        ).images[0]

    else:
        image = pipe(
            image=init_images,
            control_image=control_image,
            mask_image=init_masks,
            half_mask=half_masks,
            strength=flux_cfg.strength,
            prompt=prompt,
            height=height,
            width =width,
            num_inference_steps=flux_cfg.steps,
            guidance_scale=flux_cfg.guidance_scale,
            generator=torch.Generator().manual_seed(cfg.seed),
            output_subdir=output_subdir,
            is_grid=(cfg.flux.use_grid and not is_init),
            is_init=is_init,
            debug=False,
            low_freq_ratio=flux_cfg.low_freq_ratio,
            replace_type=flux_cfg.replace_type,
            replace_step=flux_cfg.replace_steps,
            replace_limit=flux_cfg.replace_limit,
            stop_replace_step=flux_cfg.stop_replace_steps,
        ).images[0]

    return image
