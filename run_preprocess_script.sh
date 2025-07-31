#!/bin/bash
#
# This script automates the pre-processing for multiple 3D objects.
# For each object, it performs the following steps:
# 1. Verifies that the required input files (.glb model and .txt prompt) exist.
# 2. Runs the Python rendering script to generate multi-view images.
# 3. Moves and renames the text prompt into the newly created output directory.

# --- Configuration ---
# Set the name of your dataset directory.
DATASET_NAME="MyData"

# Add the names of the objects you want to process to this list.
# Do not include the file extension.
OBJECT_NAMES=(
    "knight"
)
# ---------------------

# Define the base directory for all inputs and outputs.
BASE_DIR="./Inputs/3D/$DATASET_NAME"

# Main sanity check: Ensure the base dataset directory exists before starting.
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base dataset directory not found at '$BASE_DIR'"
    echo "Please make sure your dataset is set up correctly."
    exit 1
fi

# Loop through each object name in the array.
for obj_name in "${OBJECT_NAMES[@]}"; do
    echo "--- Processing: $obj_name ---"

    # Define the full paths for the input model and prompt files.
    INPUT_MODEL_PATH="$BASE_DIR/$obj_name.glb"
    INPUT_PROMPT_PATH="$BASE_DIR/$obj_name.txt"
    OUTPUT_DIR="$BASE_DIR/$obj_name"

    # --- Sanity Checks for this Object ---
    # 1. Check if the input model file exists.
    if [ ! -f "$INPUT_MODEL_PATH" ]; then
        echo "Warning: Model file not found for '$obj_name' at '$INPUT_MODEL_PATH'. Skipping."
        continue # Skip to the next object in the loop.
    fi

    # 2. Check if the input prompt file exists.
    if [ ! -f "$INPUT_PROMPT_PATH" ]; then
        echo "Warning: Prompt file not found for '$obj_name' at '$INPUT_PROMPT_PATH'. Skipping."
        continue # Skip to the next object in the loop.
    fi

    # --- Run Python Pre-processing Script ---
    echo "Running rendering script for '$obj_name'..."
    python preprocess.py -- \
        --object_path "$INPUT_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --radius 1.0 \
        --use_emission_shader \
        --random_camera \
        --num_renders 100 \
        --samples 256 \
        --axis_up='Y' \
        --axis_forward='-Z'

    # --- Post-processing: Move and Rename Prompt ---
    # After the script runs, check if the output directory was successfully created.
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Moving prompt file to output directory..."
        mv "$INPUT_PROMPT_PATH" "$OUTPUT_DIR/prompt.txt"
        echo "Successfully organized files for '$obj_name'."
    else
        echo "Error: Output directory '$OUTPUT_DIR' was not created by the Python script. The prompt file was not moved."
    fi

    echo "--- Finished processing $obj_name ---"
    echo "" # Add a blank line for better readability in the terminal.
done

echo "All objects processed."
