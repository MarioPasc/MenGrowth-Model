#!/bin/bash
# Benchmark DistDiff generation speed

# Path to your dataset
DATA_PATH="/media/mpascual/Sandisk2TB/research/medsyn/PathMNIST/PathMNIST.npz"

# Path to your pretrained guide model (using model_best.pth.tar)
GUIDE_MODEL_PATH="/media/mpascual/Sandisk2TB/research/medsyn/synthetic_samples/DistDiff/training_stuff/checkpoints/guide_model/model_best.pth.tar"

# Check if data path is provided as argument
if [ "$1" != "" ]; then
    DATA_PATH="$1"
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    echo "Usage: $0 [path_to_npz]"
    exit 1
fi

if [ ! -f "$GUIDE_MODEL_PATH" ]; then
    echo "Warning: Guide model not found at $GUIDE_MODEL_PATH"
    echo "Please check the path or update the script."
fi

echo "Running benchmark with dataset: $DATA_PATH"
echo "Using guide model: $GUIDE_MODEL_PATH"

# Ensure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:.

python medsyn/models/distdiff/benchmark_generation.py \
  --dataset_name pathmnist_npz \
  --data_dir "$DATA_PATH" \
  --encoder_weight_path "$GUIDE_MODEL_PATH" \
  --output_dir ./benchmark_results \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --arch open_clip_vit_b32 \
  --guidance_type transform_guidance \
  --optimize_targets global_prototype-local_prototype \
  --guidance_step 10 \
  --guidance_period 2 \
  --strength 0.2 \
  --rho 10.0 \
  --constraint_value 0.2 \
  --num_images_per_prompt 10 \
  --train_batch_size 1 \
  --gradient_checkpointing \
  --report_to none
