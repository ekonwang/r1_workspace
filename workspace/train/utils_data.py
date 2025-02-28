import jsonlines
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import json
from tqdm import tqdm

class GeomverseJsonlDataset(Dataset):
    def __init__(self, file_path, geomverse_root=None):
        """
        Args:
            file_path (str): Path to the JSONL file.
        """
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

                self.data.append(new_obj)
                if len(self.data) >= 1000:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_custom_dataset(file_path, train_split_ratio=0.8):
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
    dataset = GeomverseJsonlDataset(file_path, os.path.join(os.path.dirname(file_path), '../../..'))
    train_size = int(len(dataset) * train_split_ratio)
    test_size = len(dataset) - train_size
    
    # Split the dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # return {
    #     'train': train_dataset,
    #     'test': test_dataset
    # }
    return train_dataset, test_dataset

