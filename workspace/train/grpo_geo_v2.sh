# we explore increase Qwen2.5-2B-Instruct textually on Geometry (@Geomverse) with GRPO (Group Relative Policy Optimization @Deepseek).
# @yikunwang 3.3
export DEBUG_MODE="true"

MODEL_NAME=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/models/Qwen_Qwen2-VL-2B-Instruct
DATASET_NAME=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl
OUTPUT=./outputs/geo_v2
export LOG_PATH=${OUTPUT}/debug_log_2b.txt

rm $LOG_PATH
cd /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/workspace/train

mkdir -p $OUTPUT
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    grpo_geo_v2.py \
    --output_dir $OUTPUT \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-Geomverse-v1 \
    --save_steps 100 \
    --save_only_model true \
    --deepspeed ./local_scripts/zero3.json \
    --num_generations 6
