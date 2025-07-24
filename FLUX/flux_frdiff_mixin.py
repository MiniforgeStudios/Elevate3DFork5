import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import PIL
import numpy as np
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur

import torch.nn.functional as F

from FLUX.utils.fft import (extract_low_frequency_and_mask, 
                            extract_high_frequency, 
                            combine_latents, 
                            )

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

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


@dataclass
class FluxRefPipelineOutput(FluxPipelineOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    x_hat_t_lq_y_t_L2: List[float]
    x_hat_t_lq_y_t_cos: List[float]
    x_hat_t_hq_y_t_L2: List[float]
    x_hat_t_hq_y_t_cos: List[float]
    lq_y_t_hq_y_t_L2: List[float]
    lq_y_t_hq_y_t_cos: List[float]

class FRDiffMixin:
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        image_latent: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        half_mask: PipelineImageInput = None,
        masked_image_latents: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        low_freq_ratio: float = 0.0625,
        low_replace_str: float = 1.0, 
        output_subdir =  None,
        init_method: str = "SDEdit",
        pass_num: int = 0,
        is_grid: bool = True,
        is_init: bool = False,
        replace_type: Optional[str] = "lf",
        debug: bool = False,
        replace_step: int = 850,
        replace_limit: int = 0,
        stop_replace_step: int = 500,
        gamma: float = 0.5,
        inpaint_width: int = 128,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
    
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            mask_image_latent (`torch.Tensor`, `List[torch.Tensor]`):
                `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
                latents tensor will ge generated by `mask_image`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
                image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
                with the same aspect ration of the image and contains all masked area, and then expand that area based
                on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
                resizing to the original image size for inpainting. This is useful when the masked area is small while                the image is large and contain information irrelevant for inpainting, such as background.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
    
        Examples:
    
        Retuns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
    
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
    
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
    
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        is_strength_max = strength == 1.0
    
        # 2. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"
        
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)
    
        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    
        device = self._execution_device
    
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(timesteps, num_inference_steps, strength, device)
    
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    
        # 5. Prepare latent variables
        if control_image is None:
            num_channels_latents = self.transformer.config.in_channels // 4
        else:
            num_channels_latents = self.transformer.config.in_channels // 8
            
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )

            if control_image.ndim == 4:
                control_image = self.vae.encode(control_image).latent_dist.sample(generator=generator)
                control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                height_control_image, width_control_image = control_image.shape[2:]
                control_image = self._pack_latents(
                    control_image,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )
            num_channels_transformer = self.transformer.config.in_channels

        if init_method == 'inversion':
            latents, noise, image_latents, latent_image_ids, replace_latents = self.prepare_latents_by_inversion(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                init_image,
                timesteps,
                is_strength_max,
                gamma=gamma,
                output_subdir=output_subdir,
                debug=debug,
                inpaint_width=inpaint_width,
            )
        else:
            latents, noise, image_latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                init_image,
                latent_timestep,
                is_strength_max,
                image_latents=image_latent,
            )
        
        ori_unpacked_latents = self._unpack_latents(image_latents, height, width, self.vae_scale_factor)
    
        if is_grid:
            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
                )
    
            mask = self.prepare_mask(
                mask_condition,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                )
    
            if is_grid:
                half_mask_condition = self.mask_processor.preprocess(
                    half_mask, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
                    )
    
                half_mask = self.prepare_mask(
                    half_mask_condition,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    )
    
                half_mask = self._pack_latents(
                    half_mask.repeat(1, num_channels_latents, 1, 1),
                    half_mask.shape[0],
                    num_channels_latents,
                    2 * (int(height) // self.vae_scale_factor),
                    2 * (int(width) // self.vae_scale_factor),
                )
    
        else:
            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
                )
    
            mask = self.prepare_mask(
                mask_condition,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                )
    
        mask = self._pack_latents(
            mask.repeat(1, num_channels_latents, 1, 1),
            mask.shape[0],
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor),
        )
    
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        curr_mask = None
        
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
    
                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                
                if control_image is not None:
                    latent_model_input = torch.cat([latents, control_image], dim=2)
                else:
                    latent_model_input = latents
    
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    #attention_mask=curr_mask,
                )[0]
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                zt_prev = latents.clone() # z_t
                
                latents = self.scheduler.step(noise_pred, 
                                              t, 
                                              latents, 
                                              return_dict=False, 
                                              reference_image=None)[0] #pred z_{t-1}
    
                sigma = self.scheduler.sigmas[self.scheduler.step_index]
                x0_pred = (latents - sigma * noise_pred) / (1. - sigma)
                
                #NOTE: SDEdit
                init_latents_proper = image_latents
                init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    noise = randn_tensor(noise.shape, generator=generator, device=noise.device, dtype=noise.dtype)

                    ref_bs = init_latents_proper.size(0)
                    # Generate a random index
                    random_idx = torch.randint(low=0, high=ref_bs, size=(1,))
                    # Select the batch while keeping the batch dimension
                    init_latents_proper = init_latents_proper[random_idx]

                    init_latents_proper = self.scheduler.scale_noise(
                        init_latents_proper, torch.tensor([noise_timestep]), noise
                    )

                    if init_method == 'inversion' and is_grid:
                        init_latents_proper_unpk = self._unpack_latents(init_latents_proper, height, width, self.vae_scale_factor)
                        init_latents_proper_unpk[:, :, :, :inpaint_width] = replace_latents[i+1]
                        init_latents_proper = self._pack_latents(
                            init_latents_proper_unpk,
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            2 * (int(height) // self.vae_scale_factor),
                            2 * (int(width) // self.vae_scale_factor),
                        )
    
                    refine_image_latents = self._unpack_latents(init_latents_proper, height, width, self.vae_scale_factor)
                    if is_grid:
                        refine_image_latents = refine_image_latents[:, :, :, inpaint_width:]

                    im_low_latents, freq_mask  = extract_low_frequency_and_mask(refine_image_latents.float(), low_freq_ratio=low_freq_ratio)
                    scaled_freq_mask = F.interpolate(freq_mask, scale_factor=8, mode='nearest-exact')
                    im_low_latents  = im_low_latents.to(torch.bfloat16)
                    im_high_latents = extract_high_frequency(refine_image_latents.float(), low_freq_ratio=low_freq_ratio).to(torch.bfloat16)
    
                    if debug:
                        save_image(freq_mask[0, 0, :, :],     output_subdir / f'{height}_{width}_low_freq_mask_low_freq_ratio_{low_freq_ratio:3f}.png')
                        save_image(1 - freq_mask[0, 0, :, :], output_subdir / f'{height}_{width}_high_freq_mask_low_freq_ratio_{low_freq_ratio:3f}.png')
                        save_image(scaled_freq_mask[0, 0, :, :],     output_subdir / f'scaled_low_freq_mask_thresh_{low_freq_ratio:3f}.png')
                        save_image(1 - scaled_freq_mask[0, 0, :, :], output_subdir / f'scaled_high_freq_mask_thresh_{low_freq_ratio:3f}.png')
                        low_save_latents = (im_low_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                        image = self.vae.decode(low_save_latents, return_dict=False)[0]
                        image = self.image_processor.postprocess(image, output_type='pil')[0]
                        image.save(output_subdir / f"noised_init_image_low_freq_thresh_{low_freq_ratio}_step_{i:03}.png")
    
                        high_save_latents = (im_high_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                        image = self.vae.decode(high_save_latents, return_dict=False)[0]
                        image = self.image_processor.postprocess(image, output_type='pil')[0]
                        image.save(output_subdir / f"noised_init_image_high_freq_thresh_{low_freq_ratio}_step_{i:03}.png")
                    
                    if (i < replace_step) and (t.item() > replace_limit):
                        print(f'{i} < {replace_step}, {t.item()} > {replace_limit}')
                        if is_init:
                            latents[0] = (1 - init_mask[pass_num]) * init_latents_proper[0] + init_mask[pass_num] * latents[0]

                        latents_unpacked             = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                        init_latents_proper_unpacked = self._unpack_latents(init_latents_proper, height, width, self.vae_scale_factor)
                        noise_unpacked               = self._unpack_latents(noise, height, width, self.vae_scale_factor)
    
                        if is_grid:
                            latents_unpacked = latents_unpacked[:, :, :, inpaint_width:]
    
                        low_latents, _        = extract_low_frequency_and_mask(latents_unpacked.float(), low_freq_ratio=low_freq_ratio)
                        high_latents          = extract_high_frequency(latents_unpacked.float(),         low_freq_ratio=low_freq_ratio)
                        
                        if debug:
                            low_save_gen_latents = (low_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                            image = self.vae.decode(low_save_gen_latents.to(torch.bfloat16), return_dict=False)[0]
                            image = self.image_processor.postprocess(image, output_type='pil')[0]
                            image.save(output_subdir / f"gen_image_low_freq_thresh_{low_freq_ratio}_step_{i:03}.png")
    
                            high_save_gen_latents = (high_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                            image = self.vae.decode(high_save_gen_latents.to(torch.bfloat16), return_dict=False)[0]
                            image = self.image_processor.postprocess(image, output_type='pil')[0]
                            image.save(output_subdir / f"gen_image_high_freq_thresh_{low_freq_ratio}_step_{i:03}.png")
    
                        low_latents      = low_latents.to(torch.bfloat16)
                        high_latents     = high_latents.to(torch.bfloat16)

                        #WARN: BE CAREFUL!!!
                        if replace_type == 'swap_im_hf':
                            combined_latents = combine_latents(low_latents, im_high_latents)
                        elif replace_type == 'swap_im_lf':
                            combined_latents = combine_latents(im_low_latents, high_latents)
                        else:
                            combined_latents = latents_unpacked
    
                        full_latents = torch.zeros_like(noise_unpacked)
    
                        if is_grid:
                            full_latents[:, :, :, :inpaint_width] = init_latents_proper_unpacked[:, :, :, :inpaint_width]
                            full_latents[:, :, :, inpaint_width:] = combined_latents
                        else:
                            full_latents = combined_latents
    
                        latents = self._pack_latents(
                            full_latents,
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            2 * (int(height) // self.vae_scale_factor),
                            2 * (int(width) // self.vae_scale_factor),
                        )
    
                    if i < stop_replace_step:
                        if is_grid: 
                            if debug:
                                init_latents_proper_unpacked = self._unpack_latents(init_latents_proper, height, width, self.vae_scale_factor)
                                low_save_gen_latents = (init_latents_proper_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                                image = self.vae.decode(low_save_gen_latents.to(torch.bfloat16), return_dict=False)[0]
                                image = self.image_processor.postprocess(image, output_type='pil')[0]
                            latents[0] = (1 - init_mask[pass_num]) * init_latents_proper[0] + init_mask[pass_num] * latents[0]
                
                else:
                    pass
    
                if num_images_per_prompt > 1:
                    latents[1:] = init_latents_proper[1:]
    
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
    
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
    
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
    
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
    
                if XLA_AVAILABLE:
                    xm.mark_step()
    
        if output_type == "latent":
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            decode_latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(decode_latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type='pil')
    
            if not return_dict:
                return (image, latents)
    
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
    
        # Offload all models
        self.maybe_free_model_hooks()
    
        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
