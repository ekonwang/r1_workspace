import jsonlines
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import re


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
    pass


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

