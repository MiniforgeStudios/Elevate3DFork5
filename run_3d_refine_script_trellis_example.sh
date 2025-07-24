#!/bin/bash

# List of obj_name values
obj_names=(
    "knight"
    "bird"
    "dog_h"
    "fox"
)

GPU=0

# Loop over each obj_name
for obj_name in "${obj_names[@]}"; do
    echo "Processing object: $obj_name"

    CUDA_VISIBLE_DEVICES=$GPU python main_refactor.py --obj_name="$obj_name" --conf_name="config_trellis" --bake_mesh --device_idx=$GPU

    echo "Finished processing object: $obj_name"
    echo "-------------------------------------"
done
