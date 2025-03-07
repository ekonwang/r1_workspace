from datasets import load_dataset
from huggingface_hub import snapshot_download

# change the current directory to the workspace
import os
from utils import print_error

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
print(os.getcwd())

# # Benchmarks
# dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")
# dataset = load_dataset("di-zhang-fdu/MATH500")
# dataset = load_dataset("hendrydong/gpqa_diamond")
# dataset = load_dataset("cais/mmlu", "all")
# dataset = load_dataset("princeton-nlp/SWE-bench_Verified")

# # Alignment Dataset
# dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "Qwen/Qwen2.5-3B"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "HuanjinYao/Mulberry_qwen2vl_7b"
# model_name = "Qwen/Qwen2.5-3B-Instruct"

def snap_download_model(model_name):
    # model_name = "Qwen/Qwen2-VL-2B-Instruct"
    local_model_name = model_name.replace("/", "_")

    while True:
        try:
            model_path = snapshot_download(
                repo_id=model_name,  # The model ID on Hugging Face Hub
                # cache_dir="./.temp/models",  # Local directory to save the model
                local_dir="./.temp/models/" + local_model_name  # Specific directory for this model
            )
            break 
        except Exception as err:
            print_error(err)


def snap_download_dataset(dataset_name):
    local_dataset_name = dataset_name.replace("/", "_")

    while True:
        try:
            snapshot_download(
                repo_id=dataset_name, 
                repo_type="dataset", 
                local_dir="./.temp/datasets/" + local_dataset_name
            )
            break 
        except Exception as err:
            print_error(err)

# snap_download_model("Qwen/Qwen2.5-3B-Instruct")
# snap_download_model("Qwen/Qwen2.5-7B-Instruct")
snap_download_model("Qwen/Qwen2.5-14B-Instruct")

# from huggingface_hub import hf_hub_download
# # snapshot_download(repo_id="Jiayi-Pan/Countdown-Tasks-3to4", repo_type="dataset", local_dir="./.temp/datasets/Countdown-Tasks-3to4")
# snapshot_download(repo_id="leonardPKU/clevr_cogen_a_train", repo_type="dataset", local_dir="./.temp/datasets/clevr_cogen_a_train")
