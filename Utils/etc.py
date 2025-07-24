import shutil
import random
import re
import gc
from pathlib import Path
from typing import List, Tuple, Dict
from enum import Enum

import torch
import cv2
import numpy as np
from PIL import Image

class RefineType(Enum):
    SIGNIFICANT = 0
    MINOR = 1
    NEGLIGIBLE = 2
    SKIP = 3

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def copy_texture_files(src_texture_dir: Path, dest_texture_dir: Path, num_samples: int):
    """
    Copies all .png files from the source texture directory to the destination texture directory.
    """
    # Gather all .png files matching the pattern "train_*.png"
    png_files: List[Path] = list(src_texture_dir.glob("train_*.png"))

    total_files = len(png_files)
    sampled_files = random.sample(png_files, total_files - num_samples)

    # Move each sampled file to the destination directory
    moved_files = 0
    for png_file in sampled_files:
        try:
            dest_path = dest_texture_dir / png_file.name
            shutil.copy(str(png_file), str(dest_path))
            #print(f"Moved '{png_file.name}' to '{dest_texture_dir}'.")
            moved_files += 1
        except Exception as e:
            print(f"Failed to move '{png_file.name}': {e}")

    print(f'Moved total of {moved_files} files')


def copy_obj_file(src_obj_dir: Path, dest_mesh_dir: Path, new_name: str = "coarse.obj", decimate: bool = False):
    """
    Copies the .obj file from the source directory to the destination mesh directory and renames it.
    Assumes there is only one .obj file in the source directory.
    """
    if decimate:
        obj_files = list(src_obj_dir.glob("*_decimated.obj"))
    else:
        obj_files = list(src_obj_dir.glob("*_normalized.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj file found in {src_obj_dir}")
    src_obj = obj_files[0]
    dest_obj = dest_mesh_dir / new_name
    shutil.copy(src_obj, dest_obj)
    print(f"Copied and renamed {src_obj} to {dest_obj}")
    return dest_obj

def parse_target_angles(texture_dir: Path) -> List[Tuple[float, float]]:
    """
    Parses the target angles from texture .png filenames.
    Assumes filenames are in the format '*_yaw_pitch.png'.
    
    Returns:
        List of tuples containing (yaw, pitch) as floats.
    """
    pattern = re.compile(r'.*_(?P<yaw>-?\d+(\.\d+)?)_(?P<pitch>-?\d+(\.\d+)?)\.png$')
    target_angles = []
    for png_file in texture_dir.glob("*.png"):
        match = pattern.match(png_file.name)
        if match:
            yaw = float(match.group("yaw"))
            pitch = float(match.group("pitch"))
            target_angles.append((pitch, yaw))
            print(f"Parsed angles from {png_file.name}: yaw={yaw}, pitch={pitch}")
        else:
            print(f"Filename {png_file.name} does not match the pattern. Skipping.")
    return target_angles

def uncrop_image(image: Image.Image, zoom: float, original_size: Tuple[int, int]) -> Image.Image:
    """
    Applies zoom to the image and pads it to the original size.

    Parameters:
        image (Image.Image): The image to uncrop.
        zoom (float): Zoom factor applied to the image.
        original_size (Tuple[int, int]): Original image size (width, height).

    Returns:
        Image.Image: The uncropped and padded image.
    """
    original_width, original_height = original_size
    # Calculate new size after zoom
    new_width = int(original_width * zoom)
    new_height = int(original_height * zoom)
    # Resize the image with zoom
    zoomed_image = image.resize((new_width, new_height), Image.BICUBIC)
    # Create a new blank image with original size
    uncropped_image = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))
    # Calculate position to paste the zoomed image (centered)
    paste_x = (original_width - new_width) // 2
    paste_y = (original_height - new_height) // 2
    uncropped_image.paste(zoomed_image, (paste_x, paste_y))
    print(f"Applied zoom: {zoom} to image and padded to {original_size}")
    return uncropped_image

def extract_yaw_pitch(filename: Path) -> tuple:
    """
    Extract yaw and pitch as floats from the filename.
    Assumes that yaw and pitch are the last two underscore-separated components before the file extension.
    
    Examples:
        "texture_140_30.png" -> (140.0, 30.0)
        "texture_140.0_30.0.png" -> (140.0, 30.0)
    
    Parameters:
        filename (Path): The filename to parse.
    
    Returns:
        tuple: A tuple containing (yaw, pitch) as floats.
               Returns (None, None) if extraction fails.
    """
    name = filename.stem  # Remove extension, e.g., "texture_140_30"
    parts = name.split("_")
    if len(name.split('_')) != 4:
        yaw, pitch = map(float, name.split('_')[-2:])
        zoom = 1.
    else:
        yaw, pitch, zoom = map(float, name.split('_')[-3:])

    return (yaw, pitch)

    # if len(parts) < 2:
    #     return (None, None)
    # try:
    #     # Assuming yaw and pitch are the last two parts
    #     yaw = float(parts[-2])
    #     pitch = float(parts[-1])
    #     return (yaw, pitch)
    # except ValueError:
    #     return (None, None)

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to be within the range [0., 360).
    
    Parameters:
        angle (float): The angle to normalize.
    
    Returns:
        float: The normalized angle.
    """
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle

def move_existing_files(texture_dir: Path, old_texture_dir: Path, yaw: float, pitch: float, epsilon: float = 1):
    """
    Move texture files matching the given yaw and pitch to the old_texture_dir.
    
    Parameters:
        texture_dir (Path): Directory containing current texture files.
        old_texture_dir (Path): Directory to move old texture files to.
        yaw (float): Yaw angle to match.
        pitch (float): Pitch angle to match.
        epsilon (float, optional): Tolerance for float comparison. Defaults to 1.
    
    Returns:
        None
    """
    moved_files = 0
    target_yaw = normalize_angle(yaw)

    for file in texture_dir.glob("*.png"):
        file_yaw, file_pitch = extract_yaw_pitch(file)
        file_yaw   = normalize_angle(file_yaw)
        if file_yaw is None or file_pitch is None:
            continue  # Skip files that don't match the expected pattern
        
        # Compare yaw and pitch with tolerance
        if abs(file_yaw - yaw) < epsilon and abs(file_pitch - pitch) < epsilon:
            dest_path = old_texture_dir / file.name
            try:
                shutil.move(str(file), str(dest_path))
                print(f"Moved existing file {file.name} to {dest_path}")
                moved_files += 1
            except Exception as e:
                print(f"Failed to move file {file.name}: {e}")
    
    if moved_files == 0:
        print(f"No existing files matched yaw={yaw} and pitch={pitch} within epsilon={epsilon}")

def rename_existing_files(texture_dir: Path, yaw: float, pitch: float, refine_type: RefineType = None, epsilon: float = 1e-3):
    """
    Rename texture files matching the given yaw and pitch by prepending 'to_be_refined_' to their names.
    
    Additional Constraint:
        - Do not rename files that already start with 'refined' or 'to_be_refined_'.
    
    Parameters:
        texture_dir (Path): Directory containing current texture files.
        yaw (float): Yaw angle to match.
        pitch (float): Pitch angle to match.
        epsilon (float, optional): Tolerance for float comparison. Defaults to 1e-3.
    
    Returns:
        None
    """
    renamed_files = 0
    target_yaw = normalize_angle(yaw)

    for file in texture_dir.glob("*.png"):
        if file.name.startswith(("refined", "warn_to_be_refined", "not_important_but_to_be_refined")):
            print(f"File {file.name} already starts with prefix. Skipping.")
            continue  # Avoid renaming already refined or to-be-refined files

        file_yaw, file_pitch = extract_yaw_pitch(file)
        file_yaw = normalize_angle(file_yaw)

        if file_yaw is None or file_pitch is None:
            continue  # Skip files that don't match the expected pattern
        
        # Compare yaw and pitch with tolerance
        # WARN: Testing the effect of blending
        if abs(file_yaw - yaw) < epsilon and abs(file_pitch - pitch) < epsilon:
            # Additional Constraint: Do not rename files starting with 'refined' or 'to_be_refined_'
            
            if refine_type is None:
                new_name = f"warn_to_be_refined_{file.name}"
            elif refine_type != RefineType.SIGNIFICANT:
                new_name = f"not_important_but_to_be_refined_{file.name}"
            else:
                new_name = f"warn_to_be_refined_{file.name}"
            dest_path = texture_dir / new_name
            try:
                file.rename(dest_path)
                print(f"Renamed {file.name} to {new_name}")
                renamed_files += 1
            except Exception as e:
                print(f"Failed to rename file {file.name}: {e}")
    
    if renamed_files == 0:
        print(f"No existing files matched yaw={yaw} and pitch={pitch} within epsilon={epsilon}")
    else:
        print(f"Total renamed files: {renamed_files}")

def dilate_mask_legacy(
    gray_image: np.ndarray,
    fg_mask: np.ndarray,
    threshold_value: int = 200,
    closing_kernel_size: tuple = (5, 5),
    dilate_kernel_size: tuple = (2, 2),
    gaussian_blur_sigma: float = 5,
    final_threshold: int = 50
) -> Image.Image:
    """
    Applies dilation and morphological operations to a grayscale mask with respect to a foreground mask.

    Parameters:
    - gray_image: np.ndarray
        Grayscale image to process (2D array).
    - fg_mask: np.ndarray
        Foreground mask to constrain the dilation (2D binary array).
    - threshold_value: int, optional
        Threshold to binarize the grayscale image. Defaults to 200.
    - closing_kernel_size: tuple, optional
        Kernel size for the morphological closing operation. Defaults to (9, 9).
    - dilate_kernel_size: tuple, optional
        Kernel size for the dilation operation. Defaults to (4, 4).
    - gaussian_blur_sigma: float, optional
        Sigma value for Gaussian blurring. Defaults to 5.
    - final_threshold: int, optional
        Threshold to finalize the mask after blurring. Defaults to 50.

    Returns:
    - Image.Image
        The processed mask as a PIL Image.
    """
    # Ensure the input images are in the correct format
    if len(gray_image.shape) != 2:
        raise ValueError("gray_image must be a 2D array (grayscale).")
    if len(fg_mask.shape) != 2:
        raise ValueError("fg_mask must be a 2D array (binary mask).")

    fg_mask_binary = (fg_mask > 0).astype(np.uint8) * 255

    # Step 0: Dilate fg mask with Gaussian blur and thresholding
    # fg_mask = cv2.GaussianBlur(fg_mask_binary, (0, 0), gaussian_blur_sigma)
    # _, fg_mask_binary = cv2.threshold(fg_mask, final_threshold, 255, cv2.THRESH_BINARY)
    
    # Step 1: Threshold the grayscale image
    _, mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 3: Morphological Closing to remove small holes (CLOSING)
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

    # Step 4: Dilation to expand the mask (DILATE)
    dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, dilate_kernel, iterations=1)

    # Step 5: Gaussian Blurring to smooth the mask edges
    mask = cv2.GaussianBlur(mask, (0, 0), gaussian_blur_sigma)

    # Step 6: Final Thresholding to binarize the blurred mask
    _, mask = cv2.threshold(mask, final_threshold, 255, cv2.THRESH_BINARY)

    # Perform logical AND between the processed mask and the foreground mask
    combined_mask = cv2.bitwise_and(mask, fg_mask_binary)

    # Convert the final mask to a PIL Image
    mask_pil = Image.fromarray(combined_mask)

    return mask_pil

def dilate_mask(
    mask: np.ndarray,
    fg_mask: np.ndarray,
    closing_kernel_size: tuple = (5, 5),
    dilate_kernel_size: tuple = (5, 5),
) -> Image.Image:
    """
    Applies dilation and morphological operations to a grayscale mask with respect to a foreground mask.

    Parameters:
    - gray_image: np.ndarray
        Grayscale image to process (2D array).
    - fg_mask: np.ndarray
        Foreground mask to constrain the dilation (2D binary array).
    - threshold_value: int, optional
        Threshold to binarize the grayscale image. Defaults to 200.
    - closing_kernel_size: tuple, optional
        Kernel size for the morphological closing operation. Defaults to (9, 9).
    - dilate_kernel_size: tuple, optional
        Kernel size for the dilation operation. Defaults to (4, 4).
    - gaussian_blur_sigma: float, optional
        Sigma value for Gaussian blurring. Defaults to 5.
    - final_threshold: int, optional
        Threshold to finalize the mask after blurring. Defaults to 50.

    Returns:
    - Image.Image
        The processed mask as a PIL Image.
    """
    fg_mask_binary = (fg_mask > 0).astype(np.uint8) * 255
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, dilate_kernel, iterations=1)
    combined_mask = cv2.bitwise_and(mask, fg_mask_binary)
    mask_pil = Image.fromarray(combined_mask)

    return mask_pil

def close_mask(
    mask: np.ndarray,
    fg_mask: np.ndarray,
    closing_kernel_size: tuple = (5, 5),
) -> Image.Image:
    fg_mask_binary = (fg_mask > 0).astype(np.uint8) * 255
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    combined_mask = cv2.bitwise_and(mask, fg_mask_binary)
    mask_pil = Image.fromarray(combined_mask)

    return mask_pil

def generate_optional_schedule(num_renders: int) -> List[Dict[str, float]]:
    """
    Generates a list of camera angles based on the number of renders.

    Args:
        num_renders (int): The number of renders to generate yaw angles for each pitch.

    Returns:
        List[Dict[str, float]]: A list of dictionaries with 'yaw' and 'pitch' keys.
    """
    optional_schedule = []
    for phi in [45.0, 0.0, -45.0]:  # Set pitch angles
        for idx in range(num_renders):
            theta = (360.0 / num_renders) * idx + 15.0  # Calculate yaw angle with offset
            # Ensure theta stays within [0, 360) degrees
            theta = theta % 360.0
            optional_schedule.append({"yaw": theta, "pitch": phi})
    return optional_schedule

def generate_schedule_from_files(directory: Path, prefix: str) -> List[Dict[str, float]]:
    """
    Generates a list of camera angles by reading filenames with the specified prefix.
    
    Args:
        directory (Path): The directory containing the files.
        prefix (str): The prefix to filter files ('train' or 'test').
    
    Returns:
        List[Dict[str, float]]: A list of dictionaries with 'yaw' and 'pitch' keys.
    """
    schedule = []
    # Define the pattern to match files starting with the given prefix
    pattern = f"{prefix}_*_*.png"
    
    for file in directory.glob(pattern):
        yaw_pitch = extract_yaw_pitch(file)
        if yaw_pitch is not None:
            yaw, pitch = yaw_pitch
            schedule.append({"yaw": yaw, "pitch": pitch})
        else:
            print(f"Warning: Could not extract yaw and pitch from filename '{file.name}'")
    
    return schedule

def generate_optional_schedule_from_files(directory: Path) -> Dict[str, List[Dict[str, float]]]:
    """
    Generates optional schedules for both training and testing by reading filenames.
    
    Args:
        directory (Path): The directory containing the files.
    
    Returns:
        Dict[str, List[Dict[str, float]]]: A dictionary with keys 'train' and 'test',
                                           each mapping to their respective schedules.
    """
    train_schedule = generate_schedule_from_files(directory, "train")
    test_schedule = generate_schedule_from_files(directory, "test")
    
    return {
        "train": train_schedule,
        "test": test_schedule
    }
