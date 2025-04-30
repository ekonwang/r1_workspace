import json
import os
from utils import load_jsonl, save_jsonl

def check_img_dir_valid(img_dir):
    files = os.listdir(img_dir)
    for file in files:
        if file.startswith("selected"):
            return True
    return False


def load_sub_dir(img_dir):
    sub_paths = []
    for sub_dir in os.listdir(img_dir):
        sub_dir_path = os.path.join(img_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            if check_img_dir_valid(sub_dir_path):
                sub_paths.append(sub_dir)
    return sub_paths
            

def scavenger(img_dir, filter_img_dir, jsonl_name = "gps-mathvista-geometry-v2.jsonl"):
    sub_paths = load_sub_dir(img_dir)
    filter_sub_paths = load_sub_dir(filter_img_dir)

    jsonl_path = os.path.join(img_dir, jsonl_name)
    jsonl_list = load_jsonl(jsonl_path)
    new_jsonl = []
    assert len(jsonl_list) == len(sub_paths)

    for i, sub_name in enumerate(filter_sub_paths):
        if sub_name in sub_paths:
            new_data = jsonl_list[i]
            new_data['image_path'] = sub_name
            new_jsonl.append(new_data)
        else:
            print(f"sub_name: {sub_name} not in sub_paths")
    
    target_jsonl_path = os.path.join(filter_img_dir, jsonl_name)
    save_jsonl(new_jsonl, target_jsonl_path)

if __name__ == "__main__":
    scavenger(".temp/datasets/gps-mathvista-geometry-v2", ".temp/datasets/filter-gps-mathvista-geometry-v2", "gps-mathvista-geometry-v2.jsonl")
    scavenger(".temp/datasets/gps-olympiad-bench", ".temp/datasets/filter-gps-olympiad-bench", "gps-olympiad-bench.jsonl")
