#!/bin/bash
function rand(){
    min=$1
    max=$(($2-$min+1))

    num=$(date +%s%N)
    echo $(($num%$max+$min))
}
rnd=$(rand 29500 30500)
PORT=${PORT:-$rnd}


SCRIPT_NAME=${1:-'train.py'}
OUTPUT_DIR=${2:-'sd-model-fine-tuned-norm4'}
MODEL_PATH='../stable-diffusion-v1-5'
NUM_PROCESSES=${3:- 4}
GPU_ID=$4
extra_args=${@:5:99}

export CUDA_VISIBLE_DEVICES=$GPU_ID
COMMAND="accelerate launch --main_process_port=$PORT --num_processes $NUM_PROCESSES $SCRIPT_NAME \
         --pretrained_model_name_or_path $MODEL_PATH \
         --output_dir $OUTPUT_DIR \
         --learning_rate 1e-6 \
         --random_flip \
         --center_crop \
         --train_batch_size 1 \
         --coef_bce 0.01 \
         --coef_id 0.01 \
         --DRtune_bp_steps 2 \
         --DRtune_inference_steps 10 \
         "
echo "$COMMAND$extra_args"
eval "$COMMAND$extra_args"