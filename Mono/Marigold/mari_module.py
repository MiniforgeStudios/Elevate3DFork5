import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image

from Configs import MonoConfig  # Ensure Configs is in PYTHONPATH or same directory
from .marigold import MarigoldPipeline  # Adjust the import path as needed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import random

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def seed_all(seed: int = 0):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flip_red_channel(image: Image.Image) -> Image.Image:
    """
    Flips the red channel of the input image (255 - R).
    """
    np_image = np.array(image)
    np_image[:, :, 0] = 255 - np_image[:, :, 0]  # Invert red channel
    flipped_image = Image.fromarray(np_image)
    print("Flipped the red channel of the init_normal image.")
    return flipped_image


def setup_mari_pipeline(cfg: MonoConfig) -> MarigoldPipeline:
    """
    Initializes and returns the MarigoldPipeline based on the provided configuration.

    Parameters:
    - cfg (MariConfig): Configuration object containing model settings.

    Returns:
    - pipe (MarigoldPipeline): Initialized model pipeline.
    """
    # Determine precision
    if cfg.half_precision:
        dtype = torch.float16
        variant = None #"fp16"
        logging.info("Setting up pipeline with half precision (fp16).")
    else:
        dtype = torch.float32
        variant = None
        logging.info("Setting up pipeline with full precision (fp32).")

    # Load model components
    logging.info(f"Loading model components from checkpoint: {cfg.checkpoint}")
    unet = UNet2DConditionModel.from_pretrained(cfg.checkpoint, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(cfg.checkpoint, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(cfg.checkpoint, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.checkpoint, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(
        cfg.checkpoint, timestep_spacing=cfg.timestep_spacing, subfolder="scheduler"
    )

    # Initialize pipeline
    pipe = MarigoldPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.checkpoint,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        variant=variant,
        torch_dtype=dtype,
    )

    # Enable memory-efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logging.info("Enabled xformers memory-efficient attention.")
    except ImportError:
        logging.warning("Xformers not available. Proceeding without memory-efficient attention.")

    return pipe


def run_mari_estimation(pipe: MarigoldPipeline, cfg: MonoConfig):
    """
    Performs depth or normals estimation using the provided pipeline and configuration.

    Parameters:
    - pipe (MarigoldPipeline): Initialized model pipeline.
    - cfg (MariConfig): Configuration object containing inference settings.
    """

    # Determine modality
    normals = True if cfg.modality == 'normals' else False

    # Output directory setup
    os.makedirs(cfg.output_dir, exist_ok=True)
    if normals:
        output_dir_color = os.path.join(cfg.output_dir, "normal_colored")
        output_dir_npy = os.path.join(cfg.output_dir, "normal_npy")
    else:
        output_dir_color = os.path.join(cfg.output_dir, "depth_colored")
        output_dir_npy = os.path.join(cfg.output_dir, "depth_npy")
        output_dir_tif = os.path.join(cfg.output_dir, "depth_bw")
        os.makedirs(output_dir_tif, exist_ok=True)

    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"Output directory: {cfg.output_dir}")

    # Device configuration
    if cfg.apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
            logging.info("Using Apple Silicon (MPS) device.")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("Using CUDA device.")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    pipe = pipe.to(device)
    pipe.unet.eval()

    logging.info(f"Using device: {device}")

    # Gather input images
    rgb_filename_list = glob(os.path.join(cfg.input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images in '{cfg.input_rgb_dir}'.")
    else:
        logging.error(f"No images found in '{cfg.input_rgb_dir}'.")
        return

    # Seed setup
    if cfg.seed is None:
        seed = int(random.randint(0, 1e6))
        logging.info(f"No seed provided. Using random seed: {seed}")
    else:
        seed = cfg.seed
        logging.info(f"Using seed: {seed}")
    seed_all(seed)

    # Processing resolution
    match_input_res = cfg.output_processing_res
    if cfg.processing_res == 0 and not match_input_res:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )

    # Inference and Saving
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth/normals"):
            # Read input image
            input_image = Image.open(rgb_path).convert("RGB")
            # Predict depth or normals
            pipe_out = pipe(
                input_image,
                denoising_steps=cfg.denoise_steps,
                ensemble_size=cfg.ensemble_size,
                processing_res=cfg.processing_res,
                match_input_res=match_input_res,
                batch_size=cfg.batch_size,
                color_map=cfg.color_map,
                show_progress_bar=False,  # Disable internal progress bar
                resample_method=cfg.resample_method,
                # Additional parameters
                normals=normals,
                noise=cfg.noise,
            )

            # Extract predictions
            pred: np.ndarray = pipe_out.normal_np if normals else pipe_out.depth_np
            pred_colored: Image.Image = pipe_out.normal_colored if normals else pipe_out.depth_colored

            # Prepare filenames
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = f"{rgb_name_base}_pred"

            # Save prediction as .npy
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Overwriting existing file: {npy_save_path}")
            np.save(npy_save_path, pred)

            # Save colorized prediction
            colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}_colored.png")
            if os.path.exists(colored_save_path):
                logging.warning(f"Overwriting existing file: {colored_save_path}")
            flip_red_channel(pred_colored).save(colored_save_path)

            if not normals:
                # Save depth as 16-bit grayscale PNG
                depth_to_save = (pred * 65535.0).astype(np.uint16)
                png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
                if os.path.exists(png_save_path):
                    logging.warning(f"Overwriting existing file: {png_save_path}")
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

    logging.info("Depth/normals estimation completed successfully.")

