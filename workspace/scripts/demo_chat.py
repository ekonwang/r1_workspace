from utils import chat_vlm
import sys

# 图像插入格式：
# <img src='{file}'>

if __name__ == "__main__":
    # PROMPT_PATH = "/Users/mansionchieng/Workspaces/vlm_workspace/workspace/agent/icl_infer_prompt.txt"
    PROMPT_PATH = sys.argv[1]
    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    
    response, _ = chat_vlm(prompt)
    print(response)
