set -x

OUTPUT_DIR='./outputs/v3-grpo-14b'

export RAY_MASTER_PORT=6379
export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

ps -ef| grep ray | awk '{print $2}'|xargs kill -9
sleep 1

ray start --head  --port=$RAY_MASTER_PORT 
sleep 5

python3 -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.7 \
  --vllm_enable_sleep \
  --colocate_all_models \
  --pretrain .temp/models/Qwen_Qwen2.5-3B-Instruct \
  --remote_rm_url workspace/train/reward_func.py \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size 1 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 128 \
  --temperature 1.0 \
  --n_samples_per_prompt 6 \
  --max_epochs 1 \
  --max_samples 1000000 \
  --num_episodes 1 \
  --prompt_max_len 2048 \
  --generate_max_len 4096 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 3e-7 \
  --advantage_estimator group_norm \
  --init_kl_coef 0.01 \
  --prompt_data .temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/grpo_conversations.jsonl \
  --input_key 'prompt' \
  --label_key 'answer' \
  --apply_chat_template \
  --disable_fast_tokenizer \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps 50 \
  --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1000000 \
  --save_hf_ckpt \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward
