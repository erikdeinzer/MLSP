from torch.utils.data import Dataset
import torchvision.transforms as transforms
import kagglehub
import os
import torch
from PIL import Image
import numpy as np
from src.build.registry import TRANSFORMS, build_module

class ImageDatasetSplit(Dataset):
    def __init__(self, split, root_dir, pipeline=None):
        self.split = split
        self.root_dir = root_dir
        if pipeline is None:
            self.pipeline = transforms.ToTensor()
        else:
            tfs = [build_module(t, TRANSFORMS) for t in pipeline]
            self.pipeline = transforms.Compose(tfs)

        self.labels = None
        self.image_paths = []

    def __len__(self) -> int:
        """Get the length of the dataset.
        Returns:
            int: The number of items in the dataset.
        """
        return self.len

    def __getitem__(self, idx:int) -> dict:
        """Get an item from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            dict: A dictionary containing the image and its corresponding label.
        """
        
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        
        image = Image.open(image_path).convert("RGB")

        if self.pipeline:
            image = self.pipeline(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return {'image': image, 'label': label}
        else:
            return {'image': image}


class CustomDataset:
    def __init__(self, name, root_dir=None, **kwargs):
        self.name = name

        if root_dir is None:
            self.root_dir = kagglehub.dataset_download(name)
        else:
            self.root_dir = root_dir

        
        