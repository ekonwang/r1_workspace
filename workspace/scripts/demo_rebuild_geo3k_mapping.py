import os
import json
import argparse
from utils import load_jsonl, print_error

"""
1. load the jsonl file
2. load the train data
3. takes the 'prompt' field, find which train data's problem_text is the substring of the prompt
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--train_data_path", type=str, default="")
    return parser.parse_args()


def load_train_data(train_data_path):
    train_data = []
    subdirs = os.listdir(train_data_path)
    for subdir in subdirs:
        subdir_path = os.path.join(train_data_path, subdir)
        if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, "data.json")):
            data = json.load(open(os.path.join(subdir_path, "data.json")))
            logic_form = json.load(open(os.path.join(subdir_path, "logic_form.json")))
            data['subdir_path'] = subdir_path
            data['logic_form'] = logic_form['diagram_logic_form']
            train_data.append(data)

    return train_data


def main():
    args = parse_args()
    data = load_jsonl(args.data_path)
    train_data = load_train_data(args.train_data_path)
    
    for d in data:
        for t in train_data:
            if t['problem_text'] in d['prompt'] and d['diagram_logic_form'] == t['logic_form']:
                d['subdir_path'] = t['subdir_path']
                print_error(f"Match Found: {d['subdir_path']}")
                break