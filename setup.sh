# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install jsonlines # data utils
pip install autogen # model evaluation
pip install qwen_vl_utils torchvision
# pip install flash-attn==2.6.1 --no-build-isolation

# vLLM support 
# pip install vllm==0.7.2
# already installed 0.6.6.post1

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef


cd ./flash-attention
git switch --detach v2.7.4.post1
MAX_JOBS=4 pip install flash-attn --no-build-isolation
# run on the GPU DSW


# git clone https://github.com/ekonwang/r1_workspace_openrlhf.git
# cd r1_workspace_openrlhf && pip install -e .