from datasets import load_dataset
from huggingface_hub import snapshot_download

# Benchmarks
dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")
dataset = load_dataset("di-zhang-fdu/MATH500")
dataset = load_dataset("hendrydong/gpqa_diamond")
dataset = load_dataset("cais/mmlu", "all")
dataset = load_dataset("princeton-nlp/SWE-bench_Verified")

# Alignment Dataset
dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "Qwen/Qwen2.5-3B"
model_name = "Qwen/Qwen2.5-3B-Instruct"
local_model_name = model_name.replace("/", "_")

model_path = snapshot_download(
    repo_id=model_name,  # The model ID on Hugging Face Hub
    cache_dir="./.temp/models",  # Local directory to save the model
    local_dir="./.temp/models/" + local_model_name  # Specific directory for this model
)

from huggingface_hub import hf_hub_download
snapshot_download(repo_id="Jiayi-Pan/Countdown-Tasks-3to4", repo_type="dataset", local_dir="./.temp/datasets/Countdown-Tasks-3to4")
