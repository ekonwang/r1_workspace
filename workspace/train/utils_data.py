import jsonlines
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import re
from datasets import load_dataset


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def _mask_coordinates(tikz_code):
    """
    将 TikZ 代码中的所有具体坐标数值替换为 [MASK_VALUE]。

    参数:
        tikz_code (str): 输入的 TikZ 代码字符串。

    返回:
        str: 修改后的 TikZ 代码字符串。
    """
    # 匹配坐标的正则表达式
    coordinate_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'

    # 匹配 radius 的正则表达式
    radius_pattern = r'radius\s*=\s*(-?\d+\.?\d*)'

    # 替换函数，将匹配到的坐标或 radius 替换为 [MASK_VALUE]
    def replace_with_mask(match):
        return "([MASK_VALUE], [MASK_VALUE])" if match.group(1) else f"radius=[MASK_VALUE]"

    # 使用 re.sub 替换所有匹配的坐标
    masked_code = re.sub(coordinate_pattern, "([MASK_VALUE], [MASK_VALUE])", tikz_code)

    # 使用 re.sub 替换所有匹配的 radius
    masked_code = re.sub(radius_pattern, "radius=[MASK_VALUE]", masked_code)

    return masked_code


class GeomverseJsonlDataset(Dataset):
    def __init__(self, file_path, geomverse_root=None, sample_size=None, mask_coordinates=True):
        """
        Args:
            file_path (str): Path to the JSONL file.
        """
        assert sample_size is None or (isinstance(sample_size, int) and sample_size > 0)
        
        self.data = []
        with jsonlines.open(file_path) as reader:
            for obj in tqdm(reader):
                new_obj = dict()

                image_path = os.path.join(geomverse_root, obj['image_path'])
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
                    
                new_obj['image'] = image
                new_obj['problem'] = obj['question']
                new_obj['solution'] = f"<answer> {obj['label']} </answer>"
                new_obj['geometry'] = obj['tikz'] if not mask_coordinates else _mask_coordinates(obj['tikz'])
                new_obj['cot'] = obj["cot"]
                self.data.append(new_obj)
                
                if sample_size is not None:
                    if len(self.data) >= sample_size:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Geometry3kDataset(Dataset):
    def __init__(self, file_path, sample_size=None):
        """
        Args:
            file_path (str): Path to the JSONL file.
        """
        assert sample_size is None or (isinstance(sample_size, int) and sample_size > 0)

        self.subdirs = sorted(os.listdir(file_path))
        if sample_size is not None:
            self.subdirs = self.subdirs[:sample_size]

        self.data = []
        for subdir in self.subdirs:
            subdir_path = os.path.join(file_path, subdir)
            if os.path.isdir(subdir_path):
                self.data.append(subdir_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subdir_path = self.data[idx]
        img_path = os.path.join(subdir_path, 'img_diagram_point.png')
        data = load_json_file(os.path.join(subdir_path, 'data.json'))
        logic_form = load_json_file(os.path.join(subdir_path, 'logic_form.json'))

        problem = data['problem_text']
        choices = data['choices']
        answer = data['answer']
        diagram_logic_form = logic_form["diagram_logic_form"]
        line_instances = logic_form["line_instances"]
        dissolved_text_logic_form = logic_form["dissolved_text_logic_form"]
        
        return {
            "image": Image.open(img_path),
            "problem": problem,
            "choices": choices,
            "answer": f"<answer> {answer} </answer>",
            "diagram_logic_form": diagram_logic_form,
            "line_instances": line_instances,
            "dissolved_text_logic_form": dissolved_text_logic_form,
        }


def load_geomverse_dataset(file_path, train_split_ratio=0.8, sample_size=None, mask_coordinates=True):
    """
    Loads a custom dataset from a JSONL file and splits it into train and test sets.
    
    Args:
        file_path (str): Path to the JSONL file.
        train_split_ratio (float): Ratio of the dataset to use for training.
    
    Returns:
        train_dataset (JsonlDataset): Training subset.
        test_dataset (JsonlDataset): Testing subset.
    """

    # Root = ./.temp
    dataset = GeomverseJsonlDataset(file_path, os.path.join(os.path.dirname(file_path), '../../..'), sample_size, mask_coordinates)
    train_size = int(len(dataset) * train_split_ratio)
    test_size = len(dataset) - train_size
    
    if train_split_ratio == 1:
        return dataset
    
    # Split the dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def load_custom_dataset(file_path, train_split_ratio=0.8, sample_size=None, mask_coordinates=True):
    return load_geomverse_dataset(file_path, 
        train_split_ratio=train_split_ratio, sample_size=sample_size, mask_coordinates=mask_coordinates)


def load_geometry3k_dataset(file_path, sample_size=None):
    train_dataset = Geometry3kDataset(os.path.join(file_path, 'train'), sample_size=sample_size)
    test_dataset = Geometry3kDataset(os.path.join(file_path, 'test'), sample_size=sample_size)
    return {
        'train': train_dataset,
        'test': test_dataset,
    }


def load_mmlu_dataset(file_path='cais/mmlu', sample_size=None):
    dataset = load_dataset(file_path, trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    if sample_size is not None:
        # random sample
        train_dataset = train_dataset.shuffle(seed=7).select(range(sample_size))
        test_dataset = test_dataset.shuffle(seed=7).select(range(sample_size))
    return {
        'train': train_dataset,
        'test': test_dataset,
    }


if __name__ == "__main__":
    # test _mask_coordinates
    tikz_input = r"""
\coordinate (A) at (11.97204, 15.84669);
\coordinate (B) at (0, 0);
\coordinate (C) at (0, 19.8607);
...
\draw (-8.77681, 6.63081) -- node[left,xshift=-5mm,pos=3.59138,font=\Huge](){J}(-8.77681, 6.63081);
\draw (-15.40762, -2.146) -- node[left,xshift=-5mm,pos=6.22728,font=\Huge](){K}(-15.40762, -2.146);
"""

    # 调用函数并打印结果
    masked_output = _mask_coordinates(tikz_input)
    print(masked_output)

