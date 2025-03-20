from datasets import load_dataset
from huggingface_hub import snapshot_download

# change the current directory to the workspace
import os
from utils import print_error

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
print(os.getcwd())

def snap_download_model(model_name):
    # model_name = "Qwen/Qwen2-VL-2B-Instruct"
    local_model_name = model_name.replace("/", "_")

    while True:
        try:
            model_path = snapshot_download(
                repo_id=model_name,  # The model ID on Hugging Face Hub
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
                local_dir="./.temp/datasets/" + local_dataset_name,
            )
            break 
        except Exception as err:
            print_error(err)

# snap_download_model("Qwen/Qwen2.5-3B-Instruct")
# snap_download_model("Qwen/Qwen2.5-7B-Instruct")
# snap_download_model("Qwen/Qwen2.5-14B-Instruct")
# snap_download_model("Qwen/Qwen2.5-32B-Instruct")
# snap_download_model("Qwen/Qwen2.5-72B-Instruct")
snap_download_model("OpenRLHF/Llama-3-8b-sft-mixture")

# snapshot_download(repo_id="Jiayi-Pan/Countdown-Tasks-3to4", repo_type="dataset", local_dir="./.temp/datasets/Countdown-Tasks-3to4")
# snapshot_download(repo_id="leonardPKU/clevr_cogen_a_train", repo_type="dataset", local_dir="./.temp/datasets/clevr_cogen_a_train")
# snap_download_dataset("hiyouga/geometry3k")
snap_download_dataset("OpenRLHF/preference_dataset_mixture2_and_safe_pku")
