import os
import sys
import re
from datetime import datetime

import torch
import tiktoken
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_utils import accuracy_reward_func, format_reward_func, aux_line_reward_func, length_reward_func

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]

def reward_func(queries, prompts, labels):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    repetition_penalties = []
    # pattern = r"<\|im_start\|>\s*assistant(.*?)<\|im_end\|>"
    pattern = r"<|im_start|>assistant"

    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, label in zip(queries, prompts, labels):
            try:
                # print(re.search(pattern, query, re.DOTALL))
                # response = re.search(pattern, query, re.DOTALL).group(1).strip()

                response = query.split(pattern, 1)[1]
                answer = label

                accuracy_reward = accuracy_reward_func(response, answer, relaxed=True)
                format_reward = format_reward_func(response)
                length_reward = length_reward_func(response, max_length=900, beta=0.1)
                repetition_penalty = 0.0

                rewards.append(accuracy_reward + format_reward + length_reward)
                accuracy_rewards.append(accuracy_reward)
                format_rewards.append(format_reward)
                repetition_penalties.append(repetition_penalty)

                f.write(f"===============================================================\n")
                # f.write("Query: " + query + "\n")
                f.write("[Response]: \n\n" + response + "\n\n")
                f.write("[Answer]: \n\n" + answer + "\n\n")
                f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\tLength Reward: {length_reward}\n\n\n\n")
                f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)
                repetition_penalties.append(0.0)

    # return {
    #     "rewards": torch.tensor(rewards, dtype=torch.float32),
    #     "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
    #     "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    #     "repetition_penalties": torch.tensor(repetition_penalties, dtype=torch.float32),
    # }
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    test_completion = "Hello, world!"
    print(length_reward_func(test_completion))
    # print(len(tiktoken.encoding_for_model("gpt-4o").encode(test_completion)))
