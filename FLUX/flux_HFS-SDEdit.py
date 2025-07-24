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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
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

from FLUX.flux_refine_depth_exp_pipe import FluxInpaintPipeline
from FLUX.scheduler import FlowMatchEulerDiscreteInvertScheduler

from Utils.etc import RefineType

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


if __name__ == "__main__":
    
    pipe = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            ).to('cuda')

    pipe.scheduler = FlowMatchEulerDiscreteInvertScheduler.from_config(pipe.scheduler.config)
    
    # Define the parameter ranges
    strengths = [0.95]
    seeds = [0]
    guidance_scale = 3.5
    stop_replace_steps = [0]
    steps = 30
    down_scale = 8
    only_lq_input = True

    replace_steps = [13]
    low_freq_ratios = [0.04]

    init_method   = 'SDEdit'
    replace_types = [("swap_im_hf", "swap_im_hf")]
    # replace_types = [("swap_im_lf", "swap_im_lf")]
    # replace_types = [("swap_nothing", "swap_nothing")]

    # Define other constants
    folder_name = "example"
    exp_name    = f"test"
    is_grid     = False
    debug       = False
    two_pass    = False #True
    input_dir   = Path(f"./Inputs/2D/{folder_name}/")
    
    # Ensure output directory exists
    output_dir    = Path("./Outputs/HFS-SDEdit/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_subdir = Path(f"./Outputs/HFS-SDEdit/{folder_name}_{exp_name}")
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Attempt to save the script itself
    try:
        script_path = Path(__file__)
        shutil.copy(script_path, output_subdir / script_path.name)
    except NameError:
        print("Cannot save the script because __file__ is not defined.")


    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_path_map = {
        p.stem: p for p in input_dir.glob('*') 
        if p.suffix.lower() in image_extensions
    }
    for idx, input_prompt_path in enumerate(input_dir.glob('*.txt')) :
        im_name = input_prompt_path.stem
        
        input_img_path = image_path_map.get(im_name)

        if input_img_path:
            print(f"Processing pair: {input_img_path.name} <-> {input_prompt_path.name}")
        else:
            print(f"Warning: No matching image found for '{input_prompt_path.name}'. Skipping.")
    
        im_name = input_prompt_path.stem

        with open(input_prompt_path, 'r') as f:
            prompt = f.read()
        prompt_2 = prompt

        input_image = Image.open(input_img_path).convert('RGB')
        input_width, input_height = input_image.size
    
        # SR Settings
        input_image = center_crop(input_image, input_width - input_width % 16, input_height - input_height % 16)
        input_image.save(output_subdir / f'{im_name}_before_degradation.png')
        hq_input_image = input_image.copy()
        input_image = input_image.resize([x // down_scale for x in input_image.size], Image.Resampling.BICUBIC)
        input_image = input_image.resize([x *  down_scale for x in input_image.size], Image.Resampling.BICUBIC)
        input_image.save(output_subdir / f'{im_name}_after_degradation.png')

        width, height = input_image.size
        mask_image = Image.new("RGB", input_image.size, (255, 255, 255))

        input_image.save(output_subdir / f"{im_name}_input.png")
        mask_image.save(output_subdir  / f"{im_name}_mask.png")
        Path(output_subdir / f"{im_name}_prompt.txt").write_text(prompt)

        def decode_tensors(pipe, step, timestep, callback_kwargs, height=height, width=width, output_type="pil", low_freq_ratio=0, im_name=None, exp_name=None):
            latents = callback_kwargs["x0_pred"] #latents"]
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            image = pipe.vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type=output_type)[0]
            
            image.save(output_subdir / f"{im_name}_{exp_name}_str_{strength}_guide_{guidance_scale}_lf_ratio_{low_freq_ratio:3f}_step_{step:03}_timestep_{timestep}.png")
        
            return callback_kwargs


        for replace_type, exp_name in replace_types:
            if replace_type == "swap_nothing":
                for in_qual, in_im in [("pure_noise_input", input_image)]: # Dummy values
                    for seed in seeds:
                        for strength in strengths:
                            for low_freq_ratio in low_freq_ratios:
                                for replace_step in replace_steps:
                                    for stop_replace_step in stop_replace_steps:
                                        base_pipe = partial(
                                            pipe,
                                            prompt,
                                            prompt_2,
                                            mask_image=mask_image,
                                            num_images_per_prompt=1,
                                            height=height,
                                            width=width,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=steps,
                                            max_sequence_length=512,
                                            callback_on_step_end_tensor_inputs=["x0_pred"],
                                            output_subdir=output_subdir,
                                            is_grid=is_grid,
                                            debug=debug,
                                            replace_step=replace_step,
                                            stop_replace_step=stop_replace_step,
                                        )

                                        if debug:
                                            decode_tensors_partial = partial(decode_tensors, low_freq_ratio=low_freq_ratio, im_name=folder_name, exp_name=exp_name)
                                        else:
                                            decode_tensors_partial = None

                                        image  = base_pipe(image=in_im, 
                                                           strength=strength, 
                                                           low_freq_ratio=low_freq_ratio,
                                                           callback_on_step_end=decode_tensors_partial, 
                                                           replace_type=replace_type,
                                                           replace_step=replace_step,
                                                           generator=torch.Generator("cuda").manual_seed(seed),
                                                           ).images[0]

                                        image.save(output_subdir / f"output_{exp_name}_seed_{seed}_{im_name}_{in_qual}_{exp_name}_down_{down_scale}_sdedit_{strength}_replace_step_{replace_step}_guide_{guidance_scale}_lf_ratio_{low_freq_ratio:3f}.png")
            else:
                for in_qual, in_im in [("LQ_input", input_image), ("HQ_input", hq_input_image)]:
                    if only_lq_input and (in_qual == "HQ_input"):
                        continue
                    for seed in seeds:
                        for strength in strengths:
                            for low_freq_ratio in low_freq_ratios:
                                for replace_step in replace_steps:
                                    for stop_replace_step in stop_replace_steps:
                                        base_pipe = partial(
                                            pipe,
                                            prompt,
                                            prompt_2,
                                            mask_image=mask_image,
                                            num_images_per_prompt=1,
                                            height=height,
                                            width=width,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=steps,
                                            max_sequence_length=512,
                                            callback_on_step_end_tensor_inputs=["x0_pred"],
                                            output_subdir=output_subdir,
                                            is_grid=is_grid,
                                            debug=debug,
                                            replace_step=replace_step,
                                            stop_replace_step=stop_replace_step,
                                        )

                                        if debug:
                                            decode_tensors_partial = partial(decode_tensors, low_freq_ratio=low_freq_ratio, im_name=folder_name, exp_name=exp_name)
                                        else:
                                            decode_tensors_partial = None

                                        image  = base_pipe(image=in_im, 
                                                           strength=strength, 
                                                           low_freq_ratio=low_freq_ratio,
                                                           callback_on_step_end=decode_tensors_partial, 
                                                           replace_type=replace_type,
                                                           replace_step=replace_step,
                                                           init_method=init_method,
                                                           generator=torch.Generator("cuda").manual_seed(seed),
                                                           ).images[0]

                                        image.save(output_subdir / f"{im_name}_output_{exp_name}_seed_{seed}_{in_qual}_down_{down_scale}_sdedit_{strength}_replace_step_{replace_step}_guide_{guidance_scale}_lf_ratio_{low_freq_ratio:3f}.png")

