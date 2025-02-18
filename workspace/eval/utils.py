import json
import os, sys
from tqdm import tqdm
from datetime import datetime


def mk_pbar(iterable, ncols=80, **kwargs):
    # check if iterable
    if not hasattr(iterable, '__iter__'):
        raise ValueError("Input is not iterable.")
    return tqdm(iterable, ncols=ncols, **kwargs)


def mk_len_pbar(ncols=80, **kwargs):
    return tqdm(ncols=ncols, **kwargs)


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(func, data)
