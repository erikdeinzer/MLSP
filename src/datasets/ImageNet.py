from .ImageDataset import CustomDataset, ImageDatasetSplit
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

import pandas as pd
from src.build.registry import DATASETS
import torch
from PIL import Image

@DATASETS.register_module()
class ImageNetTrainSplit(ImageDatasetSplit):
    """ImageNet Tiny Dataset for image classification.
    
    Should be downloaded from Kaggle: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
    """
    def __init__(self, **kwargs):
        super().__init__(split='train',**kwargs)
        
        self.dir = os.path.join(self.root_dir, 'tiny-imagenet-200', 'train')        
    
        self.ds = ImageFolder(root=self.dir, transform=self.transform)
        self.class_names = self.ds.classes
        self.class_to_idx= {syn: idx for idx, syn in enumerate(self.class_names)}

        self.num_classes = len(self.class_names)
        self.len = len(self.ds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):        
        img, label = self.ds[idx]

        return {'image': img, 'label': label}

class ImageNetValSplit(ImageDatasetSplit):
    def __init__(self, cls_to_idx:dict, **kwargs):
        super().__init__(split='val', **kwargs)
        
        self.dir = os.path.join(self.root_dir, 'tiny-imagenet-200', self.split)
        self.split_file = os.path.join(self.dir, f"{self.split}_annotations.txt")
        df = pd.read_csv(
            self.split_file, 
            sep="\t",       # tab-separated
            header=None,    # no header row in the file
            names=['Filename', 'label', 'x', 'y', 'width', 'height'])
        self.cls_to_idx = cls_to_idx
        df['label_idx'] = df['label'].map(cls_to_idx)

        # get image paths and joing with "images/"
        self.image_paths = df['Filename'].apply(lambda x: os.path.join(self.dir, 'images', x)).tolist()
        self.labels = df['label_idx'].tolist()  
        self.len = len(self.image_paths)

        self.num_classes = len(set(self.labels))
        self.class_names = list(set(self.labels))

class ImageNetTestSplit(ImageDatasetSplit):
    def __init__(self, **kwargs):
        super().__init__(split='test', **kwargs)
        
        # Get all image paths from the test directory (no annotation file present)
        self.dir = os.path.join(self.root_dir, 'tiny-imagenet-200', self.split, 'images')
        self.image_paths = []
        for root, _, files in os.walk(self.dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))


        self.len = len(self.image_paths)



@DATASETS.register_module()
class ImageNetDataset(CustomDataset):
    """ImageNet Tiny Dataset for image classification.
    
    Should be downloaded from Kaggle: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
    """
    def __init__(self, name='akash2sharma/tiny-imagenet', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.root_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
        self.train_data = ImageNetTrainSplit(root_dir = self.root_dir, **kwargs)
        self.val_data = ImageNetValSplit(root_dir = self.root_dir, cls_to_idx=self.train_data.class_to_idx, **kwargs)
        self.test_data = ImageNetTestSplit(root_dir = self.root_dir, **kwargs)
