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
    
    if reward == 0.0:
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


def reward_func(queries, prompts, labels):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    aux_rewards = []
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
                aux_reward = aux_line_reward_func(response)
                repetition_penalty = 0.0

                rewards.append(accuracy_reward + format_reward + aux_reward)
                accuracy_rewards.append(accuracy_reward)
                format_rewards.append(format_reward)
                aux_rewards.append(aux_reward)
                repetition_penalties.append(repetition_penalty)

                f.write(f"===============================================================\n")
                f.write("[Response:] " + response + "\n")
                f.write("[Answer:] " + answer + "\n")
                f.write(f"[Accuracy Reward]: {accuracy_reward}\t[Format Reward]: {format_reward}\t[Auxiliary Reward]: {aux_reward}\n\n\n\n")
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
    response = r"""
<think>
To solve for the measure of angle 3, let's first summing up the information received in the problem:

1. Measure of angle M is 31 degrees.
2. Measure of angle 1 is the same as the measure of angle Q (given as 22 degrees).
3. Measure of angle 2 is the same as the measure of angle L (which is also given as 31 degrees).
4. Since line MN is perpendicular to both line LM and PQ, we know that each of the four angles formed by the intersection of the two lines (let’s call them angle 1, 2, 3, and 4) are right-angled and equal, each measuring 90 degrees.

However, angle 3 is also part of a larger context where angle 3 corresponds to the remaining angle when considering the sum of angles around a point or specific configurations provided in the problem. As MN and PQ are considered perpendicular and are made up of angle 1 (22 degrees) and angle 2 (31 degrees), angle 3 would be the supplementary angle to angle 1 when combined together.

The sum of angle 1 and 2 is: 22 + 31 = 53 degrees.

Since the total around a point would normally include 90 degrees for perpendicular lines, and considering angle 1 and 2 already consumed 53 degrees, the remaining complementary angle (to complete the perpendicular line measure) is: 90 - (90 - 53) = 90 - 53 = 37 degrees. But since angle 3 must align with these perpendicular properties, we now confirm if these complementary angles could shift by properties we saw:  

Since we are looking at the constructed lines and knowing perpendiculars, the 90 - remaining could apply to see the direct complementary angle to external line configurations. Given it must fit a summing up accurately m\angle 3 must fit best as 99.
</think>
<answer> C </answer>
"""
    answer = '<answer> 70 </answer>'
    print(format_reward_func(completion=response))
