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
To solve this problem, I need to first identify the type of triangle we are dealing with based on the lengths of the sides provided: J, K, and M. The given side lengths are 2.1, 1.3, and 1.9. Let's start with determining the lengths.

1. **Check for right triangle using Pythagorean Theorem:**
   - Hypotenuse ² = Leg₁² + Leg₂²
   - 2.1² = 1.3² + 1.9²
   - 4.41 = 1.69 + 3.61
   - 4.41 ≠ 5.3
   - Hence, it is not a right triangle.

2. **Check for isosceles triangle using Pythagorean Postulate:**
- If 2.1 is a base, then to see if it could be an isosceles triangle, we check if the other two sides are equal.

3. **Check lengths 1.3 and 1.9**
   - They don't equal each other.
   - Since 1.3² + 1.9² (5.3) ≠ 2.1² (4.41), this does not satisfy the Postulate of equal sides.

Based on the property checks, it does not represent a standard right or isosceles triangle either. However, there is a different way to solve this using trigonometry.

4. **Using trigonometry (Sine Rule on Triangle JKM):**
   - sin(x) = \(\frac{opposite}{hypotenuse}\)
   - Let's determine the acute angle x at J with respect to side KM (hypotenuse).

5. Use the Law of Cosines:
   - Law of Cosines: \( a² = b² + c² - 2bc\cos(A) \)
   - Let's apply it with sides k = 1.9, j = 1.3, and m = 2.1.
   - \( 2.1² = 1.9² + 1.3² - 2 \cdot 1.9 \cdot 1.3 \cdot \cos(x) \)
   - \( 4.41 = 3.61 + 1.69 - 4.34 \cdot \cos(x) \)
   - \( 4.41 = 5.30 - 4.34 \cdot \cos(x) \)
   - \( 4.41 - 5.30 = -4.34 \cdot \cos(x) \)
   - \(-0.89 = -4.34 \cdot \cos(x) \)
   - Dividing by -4.34, \(\cos(x) = \frac{0.89}{4.34} \approx 0.20\) 
   - Hence, \( \cos^{-1}(0.20) \approx 79.3 \)

Rounding 79.3 to the nearest degree gives us 79.

Therefore, the correct answer is *not* in the given options.

<think> Despite the attempt and the analysis, none of the provided options (43, 53, 63, 73) fit the calculated angle, which was approximately 79°. As the problem demands re-checking of logic and options provided. </think>

<answer>69.8</answer><|im_end|>
"""
    answer = '<answer> 70 </answer>'
    print(accuracy_reward_func(completion=response, answer=answer))
