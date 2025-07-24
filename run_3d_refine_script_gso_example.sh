#!/bin/bash

# List of obj_name values
obj_names=(
    "Breyer_Horse_Of_The_Year_2015"
    "Jansport_School_Backpack_Blue_Streak"
    "Nickelodeon_Teenage_Mutant_Ninja_Turtles_Raphael"
    "Ortho_Forward_Facing_QCaor9ImJ2G"
)

GPU=0

# Loop over each obj_name
for obj_name in "${obj_names[@]}"; do
    echo "Processing object: $obj_name"

    CUDA_VISIBLE_DEVICES=$GPU python main_refactor.py --obj_name="$obj_name" --conf_name="config_gso" --bake_mesh --device_idx=$GPU

    echo "Finished processing object: $obj_name"
    echo "-------------------------------------"
done
