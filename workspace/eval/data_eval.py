import os
import json
import os
import sys
import re
from datasets import load_dataset
from datetime import datetime
from utils import load_jsonl, save_jsonl, print_error
# set up the agent
# MAX_REPLY = 10
# llm_config={"cache_seed": None, "config_list": [{"model": "Qwen/Qwen2-VL-72B-Instruct", "temperature": 0.0, "api_key": "sk-wykivevwxqqrfihqaeuiqyexnzzugnvaorzmjxtfcghzrvox", "base_url": "https://api.siliconflow.cn/v1"}]}

llm_config={"cache_seed": None, "config_list": [{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "temperature": 0.0, "api_key": "sk-gjezftinzvhzoogekwilcnydixgooycpezqemudmnttqbycj", "base_url": "https://api.siliconflow.cn/v1"}]}

from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
)
from autogen.oai.client import OpenAIWrapper


def chat_vlm(prompt: str, history_messages = None, retry_times: int = 10):
    interval = 1
    for i in range(retry_times):
        try:
            if history_messages is None:
                history_messages = []
            clean_messages = history_messages + [{"role": "user", "content":  prompt}]
            dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
            
            client = OpenAIWrapper(**call_config)
            response = client.create(
                messages=dirty_messages,
                timeout=600,
            )
            messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
            return response.choices[0].message.content, messages
        except Exception as e:
            if 'limit' in str(e):
                sleep(interval)
                interval = min(interval * 2, 60)
            print_error(e)
            if i >= (retry_times - 1):
                raise e

# reply, messages = chat_gpt4o("Could you please give me a list of all the countries in the world?")
# print(reply)
def eval_question(question: str):
    reply, messages = chat_gpt4o(question)
    return reply


def _eval_aime_(element: dict):
    question = element['Question']
    answer = element['Answer']

    prompt = f"""
Here is a problem from AIME, please solve the problem step by step. You need to give the final answer in the format of "The answer is xxxx" in the end.

```
{question}
```
"""
    reply = eval_question(prompt)
    pattern = r'-?\d+\.?\d*'
    results = re.findall(pattern, reply)

    if len(results) == 0:
        print_error(f"No answer found in the reply: {reply.replace('\n', '\\n')}")
        return {'flag': 0, 'reply': reply, 'prompt': prompt, "answer": answer}
    result = results[-1]
    if float(result) != float(answer):
        print_error(f"The answer is incorrect: {result} != {answer}")
        return {'flag': 0, 'reply': reply, 'prompt': prompt, "answer": answer}
    return {'flag': 1, 'reply': reply, 'prompt': prompt, "answer": answer}


def eval_dataset(dataset, output_path, verbose: bool = False):
    tot_acc = 0
    tot_eval = 0
    all_eval_data = {}

    if os.path.exists(output_path):
        all_eval_data = load_jsonl(output_path)
        all_eval_data = {element['id']: element for element in all_eval_data}

    for element_id in range(len(dataset)):
        element = dataset[element_id]
        eid = element['ID']

        if eid in all_eval_data:
            print_error(f'{eid} already inference, skip.')

        try:
            eval_dict = _eval_aime_(element)
        except Exception as err:
            if 'keyboard' in str(err).lower():
                raise err 
            print_error(err)

        eval_dict['id'] = eid
        tot_acc += eval_dict['flag']
        tot_eval += 1
        if verbose:
            print(f"\n" * 3)
            print("=" * 40)
            print(f"Question: {eval_dict['prompt']}")
            print(">" * 20)
            print(f"Answer: {eval_dict['reply']}")
            print(">" * 20)
            print(f"Ground Truth: {eval_dict['answer']}")
            print(f'{tot_acc / tot_eval * 100:.2f} ({tot_acc:6d}/{tot_eval:6d})')
        
        # all_eval_data.append(eval_dict)
        all_eval_data[eid] = eval_dict
        save_jsonl(list(all_eval_data.values()), output_path)

    return tot_acc / tot_eval
            

dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")['train']
eval_dataset(dataset, '.temp/outputs/AIME/DeepSeek-R1-Distill-Qwen-7B.jsonl',  True)
