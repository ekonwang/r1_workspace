import os
import json
import argparse
from utils import load_jsonl, print_error, save_jsonl

"""
Geo3k conversations.jsonl has no image index, the script need to find the corresponding subdir for each data item, in order to recover the image info.

1. load the jsonl file
2. load the train data
3. takes the 'prompt' field, find which train data's problem_text is the substring of the prompt
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".temp/datasets/intergpt_geometry3k/conversations.jsonl")
    parser.add_argument("--train_data_path", type=str, default=".temp/datasets/intergpt_geometry3k/train")
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
            data['point_positions'] = logic_form['point_positions']
            train_data.append(data)

    return train_data


def main():
    args = parse_args()
    data = load_jsonl(args.data_path)
    train_data = load_train_data(args.train_data_path)

    new_data_path = args.data_path.replace('.jsonl', '_map.jsonl')
    
    new_data = []
    for d in data:
        __flag = 0
        for t in train_data:
            if t['problem_text'] in d['prompt'] and d['diagram_logic_form'] == t['logic_form']:
                d['subdir_path'] = t['subdir_path']
                d['point_positions'] = t['point_positions']
                print(f"Match Found: {d['subdir_path']}")
                new_data.append(d)
                __flag = 1
                break
        
        if not __flag:
            print_error('No Match Found!')
    
    save_jsonl(new_data, new_data_path)
    

if __name__ == "__main__":
    main()
