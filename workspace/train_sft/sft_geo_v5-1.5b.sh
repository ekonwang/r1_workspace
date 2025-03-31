# we explore increase Qwen2.5-2B-Instruct textually on Geometry (@Geomverse) with GRPO (Group Relative Policy Optimization @Deepseek).
# @yikunwang 3.3
# We apply cpu offload, gradient checkpointing etc. to avoid OOM.

export DEBUG_MODE="true"

# MODEL_NAME=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/models/Qwen_Qwen2.5-3B-Instruct
MODEL_NAME=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/models/Qwen_Qwen2.5-1.5B-Instruct
DATASET_NAME=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/datasets/0328_mix/data.jsonl
OUTPUT=./outputs/sft_geo_v5-1.5b
RUN_NAME=Qwen2.5-1.5B-SFT-Geomverse-v5
export LOG_PATH=${OUTPUT}/debug_log_1.5b.txt

rm $LOG_PATH
cd /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/workspace/train_sft

mkdir -p $OUTPUT
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12333" \
    sft_geo_v5.py \
    --output_dir $OUTPUT \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-7 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed ./local_scripts/zero3_offload-tp4.json \
    --use_sft
