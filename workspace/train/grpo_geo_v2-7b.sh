# we explore increase Qwen2.5-2B-Instruct textually on Geometry (@Geomverse) with GRPO (Group Relative Policy Optimization @Deepseek).
# @yikunwang 3.3
# We apply cpu offload, gradient checkpointing etc. to avoid OOM.

export DEBUG_MODE="true"
WORKSPACE=$(pwd)
echo $WORKSPACE

# MODEL_NAME=${WORKSPACE}/.temp/models/Qwen_Qwen2.5-3B-Instruct
MODEL_NAME=${WORKSPACE}/.temp/models/Qwen_Qwen2.5-7B-Instruct
DATASET_NAME=${WORKSPACE}/.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl
OUTPUT=./outputs/geo_v2-7b
RUN_NAME=Qwen2-VL-2B-GRPO-Geomverse-v1
export LOG_PATH=${OUTPUT}/debug_log_7b.txt

rm $LOG_PATH
cd ${WORKSPACE}/workspace/train

mkdir -p $OUTPUT
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12335" \
    grpo_geo_v2.py \
    --output_dir $OUTPUT \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed ./local_scripts/zero3_offload-tp4.json \
    --num_generations 6
