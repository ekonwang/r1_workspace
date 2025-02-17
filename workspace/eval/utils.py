import json

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
