import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]


def extract_answer_with_tags(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response == None:
        print("Failed to extract answer from completion: ", completion.replace("\n", ""))
        return 0.0

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
    
    # if reward == 0.0:
    #     # relaxed constraint
    #     try:
    #         answer_parsed_floats = parse_float(content)
    #         gold_parsed_float = parse_float(sol)[-1]
    #         for answer_parsed_float in answer_parsed_floats:
    #             if abs(answer_parsed_float - gold_parsed_float) < 2e-2 * abs(gold_parsed_float):
    #                 reward = 1.0
    #                 break
    #     except Exception:
    #         pass


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
    match = re.match(pattern, completion, re.DOTALL)
    return 0.5 if match else 0.0


def aux_line_reward_func(completion, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?<auxiliary>.*?</auxiliary>.*?</think>.*?<answer>.*?</answer>"
    match = re.match(pattern, completion, re.DOTALL)
    return 0.5 if match else 0.0


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

                accuracy_reward = accuracy_reward_func(response, answer)
                format_reward = format_reward_func(response)
                repetition_penalty = 0.0

                rewards.append(accuracy_reward + format_reward)
                accuracy_rewards.append(accuracy_reward)
                format_rewards.append(format_reward)
                repetition_penalties.append(repetition_penalty)

                f.write(f"===============================================================\n")
                f.write("Query: " + query + "\n")
                f.write("Response: " + response + "\n")
                f.write("Answer: " + answer + "\n")
                f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
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
    Q = r"""
<|im_start|>assistant
Let's analyze the given TikZ code. The rectangle is defined with vertices at points \(A (0, 0)\), \(B (10, 0)\), \(C (10, 24)\), and \(D (0, 24)\). The problem asks for the area of the yellow rectangle, which is the entire rectangle in this case since no part of the rectangle is filled in a different color or altered.

The area \(A\) of a rectangle is given by the formula:

\[ A = \text{length} \times \text{width} \]

From the TikZ code:
- The length of the rectangle is 10 (the horizontal distance between \(A\) and \(B\)).
- The width of the rectangle is 24 (the vertical distance between \(A\) and \(D\)).

Substituting these values into the area formula:

\[ A = 10 \times 24 \]

Calculating this product:

\[ A = 240 \]

Thus, the area of the yellow rectangle is 240 square units.

<think> The given TikZ code provides the coordinates of the vertices of a rectangle, with the coordinates suggesting the length is 10 units and the width is 24 units. The area of the rectangle is the product of its length and width. </think>
<answer>240.00</answer><|im_end|>
Response: Let's analyze the given TikZ code. The rectangle is defined with vertices at points \(A (0, 0)\), \(B (10, 0)\), \(C (10, 24)\), and \(D (0, 24)\). The problem asks for the area of the yellow rectangle, which is the entire rectangle in this case since no part of the rectangle is filled in a different color or altered.

The area \(A\) of a rectangle is given by the formula:

\[ A = \text{length} \times \text{width} \]

From the TikZ code:
- The length of the rectangle is 10 (the horizontal distance between \(A\) and \(B\)).
- The width of the rectangle is 24 (the vertical distance between \(A\) and \(D\)).

Substituting these values into the area formula:

\[ A = 10 \times 24 \]

Calculating this product:

\[ A = 240 \]

Thus, the area of the yellow rectangle is 240 square units.

<think> The given TikZ code provides the coordinates of the vertices of a rectangle, with the coordinates suggesting the length is 10 units and the width is 24 units. The area of the rectangle is the product of its length and width. </think>
<answer>240.00</answer>
"""
    Q1 = r"""
<|im_start|>assistant
Hello, world! <answer>99.2</answer><\|im_end\|>
"""
    queries = [Q, Q1]
    prompts = ["Hello, world!", "Hello, world!"]
    labels = ["240", "99.9"]
    print(reward_func(queries, prompts, labels))
