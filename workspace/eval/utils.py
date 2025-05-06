import json
import os, sys
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track, Progress
from time import sleep
from utils_execution import SandboxCodeExecutor
from PIL import Image

# llm_config={"cache_seed": None, "config_list": [{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "temperature": 0.0, "api_key": "sk-gjezftinzvhzoogekwilcnydixgooycpezqemudmnttqbycj", "base_url": "https://api.siliconflow.cn/v1"}]}
llm_config={"cache_seed": None, "config_list": [{"model": "Pro/deepseek-ai/DeepSeek-V3", "temperature": 0.0, "api_key": "sk-gjezftinzvhzoogekwilcnydixgooycpezqemudmnttqbycj", "base_url": "https://api.siliconflow.cn/v1"}]}

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
            
            client = OpenAIWrapper(**llm_config)
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


def mk_pbar(iterable, ncols=80, **kwargs):
    # check if iterable
    if not hasattr(iterable, '__iter__'):
        raise ValueError("Input is not iterable.")
    return tqdm(iterable, ncols=ncols, **kwargs)


# def mk_len_pbar(ncols=80, **kwargs):
def mk_len_pbar(iterable, func, **kwargs):
    # use rich progress bar
    # return track(list(range(total)), **kwargs)
    with Progress(**kwargs) as progress:
        # Create a task with an initial description
        task_id = progress.add_task("[red]Processing...", total=len(iterable))
        for i, elem in enumerate(iterable):
            results = func(elem)
            progress.update(task_id, advance=1)
            yield results


def generate_uuid():
    return str(uuid4())


def print_error(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(f"\033[91m\033[1m{message}\033[0m")


def load_jsonl(path):
    data = []
    with open(path, "rt") as f:
        for line in mk_pbar(f):
            data.append(json.loads(line.strip()))
    return data

def load_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def save_jsonl(data, path, mode='w', use_tqdm=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        if use_tqdm:
            data = mk_pbar(data)
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wt") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def tag_join(valid_tags, sep=','):
    return sep.join([str(tag) for tag in valid_tags])


def mean_list(data_list):
    data_list = [float(d) for d in data_list if d is not None]
    return sum(data_list) / len(data_list)


def log_print(*content, **kwargs):
    # set datetime timezone to Shanghai.
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    content = [f'[{datetime.datetime.now()}]'] + list(content)
    print(*content, **kwargs)
    sys.stdout.flush()


def json_print(data):
    print(json.dumps(data, ensure_ascii=False, indent=4))


def multithreading(func, thread_num=8, data=None):
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(func, data)

def code_to_image(code: str):
    executor = SandboxCodeExecutor()
    results = executor.execute(code)
    if results[0]:
        raise Exception(f'{results[1]}')

    results = results[2]
    image_path = results[0]
    image = Image.open(image_path)
    return image

    
if __name__ == '__main__':
    # pbar = mk_len_pbar(total=100, description='test')
    # for i in range(100):
    #     next(pbar)
    #     pbar.description = f'test: {i}'
    #     # pbar.set_description(f'test: {i}')
    #     # pbar.track(description=f'test: {i}')
    #     time.sleep(0.1)

    # from rich.progress import Progress
    # import time

    # with Progress() as progress:
    #     # Create a task with an initial description
    #     task_id = progress.add_task("[red]Processing...", total=100)

    #     for i in range(100):
    #         time.sleep(0.05)  # Simulate work
    #         if i == 50:
    #             # Update the description halfway
    #             progress.update(task_id, description="[green]Halfway done!")
    #         progress.update(task_id, advance=1)

    jsonl_data = load_jsonl('workspace/eval/temp-ob.jsonl')
    data0 = jsonl_data[1]
    image = code_to_image(data0['code'])
    image.show()
