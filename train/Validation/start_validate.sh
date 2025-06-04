#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=4,5

models_dir="/map-vepfs/yubo/One-Shot-CFT/output_models_math_0604_0/v1-20250604-185543"
summary_path="../../Validation/validation_summary.txt"

models_dir_name=$(basename "$models_dir")
summary_parent_dir=$(dirname "$summary_path")

for checkpoint_dir in ${models_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then

        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        if [ "$checkpoint_num" -ge 1 ] && [ "$checkpoint_num" -lt 100 ]; then

            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"

            bash validate_single.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num} as it's >= 100"
        fi
    fi

done