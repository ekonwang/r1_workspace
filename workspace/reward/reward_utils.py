import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from transformers import AutoTokenizer

__workspace_path = os.path.dirname(os.path.abspath(__file__)) + '/..'
__text_tokenizer = AutoTokenizer.from_pretrained(__workspace_path + "/tokenizer")

def __get_tokens(text):
    return __text_tokenizer.encode(text)


def extract_answer_with_tags(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def accuracy_reward_func(completion, answer, relaxed=False):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response == None:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    try:
        sol = re.search(r"<answer>(.*?)</answer>", sol).group(1).strip()
    except:
        sol = sol.strip()
    gold_parsed = parse(sol)
    
    # numeric answer
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass


    # numeric answer
    def parse_float(parsed):
        # 用 re 提取所有 
        pattern = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        return [float(p) for p in re.findall(pattern, parsed)]
    
    if reward == 0.0 and relaxed:
        # relaxed constraint
        try:
            answer_parsed_floats = parse_float(content)
            gold_parsed_float = parse_float(sol)[-1]
            for answer_parsed_float in answer_parsed_floats:
                if abs(answer_parsed_float - gold_parsed_float) < 2e-2 * abs(gold_parsed_float):
                    reward = 1.0
                    break
        except Exception:
            pass


    if sol in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] and reward == 0.0:
        # Option A, B, C, D
        try:
            # content is 
            content = response.strip().upper()
            if sol == content:
                reward = 1.0
        except:
            print("Failed to parse gold solution: ", sol)

    return reward


def format_reward_func(completion, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    match = re.match(pattern, completion.strip(), re.DOTALL)
    return 0.5 if match else 0.0


def aux_line_reward_func(completion, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?<auxiliary>.*?</auxiliary>.*?</think>.*?<answer>.*?</answer>"
    match = re.match(pattern, completion.strip(), re.DOTALL)
    return 0.5 if match else 0.0


def aux_line_reward_v2_func(completion, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<auxiliary>.*?</auxiliary>"
    match = re.findall(pattern, completion.strip(), re.DOTALL)
    # if match is not empty, return 1.0, otherwise return 0.0
    return 0.5 if match else 0.0


def length_reward_func(completion: str, max_length=900, beta:float=0.5, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    tokenized_length = len(__get_tokens(completion))
    return (min(tokenized_length, max_length) / max_length) * beta


if __name__ == "__main__":
    test_completion = "Hello, world!"
    # print(len(tiktoken.encoding_for_model("gpt-4o").encode(test_completion)))
    print(length_reward_func(test_completion))
