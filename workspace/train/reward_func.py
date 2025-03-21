import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]


def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None


def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    answer_parsed = content
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    return 0.5 if matches else 0.0


def reward_func(queries, prompts):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    repetition_penalties = []
    pattern = r"<\|im_start\|>\s*assistant(.*?)<\|im_end\|>"

    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt in zip(queries, prompts):
            try:
                query = re.sub(r"\s*<IMG_CONTEXT>\s*", "", query)
                query = re.sub(r"<img>\s*</img>", " <image>", query)
                query = re.sub("</s>", "", query)
                response = re.search(pattern, query, re.DOTALL).group(1).strip()
                answer = prompt["answer"]

                accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
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

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
        "repetition_penalties": torch.tensor(repetition_penalties, dtype=torch.float32),
    }
