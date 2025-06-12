from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
import numpy as np

from src.datasets.ImageDataset import ImageDatasetSplit, CustomDataset
from src.build.registry import DATASETS

class EuroSATSplit(ImageDatasetSplit):
    """EuroSAT Dataset for satellite image classification.
    
    Should be downloaded from Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
    """
    def __init__(self,  **kwargs):
        
        super().__init__(**kwargs)
        self.split_file = os.path.join(self.root_dir, f"{self.split}.csv")
        df = pd.read_csv(self.split_file)

        self.image_paths = df['Filename'].tolist()
        self.labels = df['Label'].tolist()  
        self.len = len(self.image_paths)

        self.num_classes = len(np.unique(self.labels))

        self.class_names = df['ClassName'].unique()
        self.class_names = np.unique(self.labels).tolist()

@DATASETS.register_module()
class EuroSATDataset(CustomDataset):
    """EuroSAT Dataset for satellite image classification.
    
    Should be downloaded from Kaggle: https://www.kaggle.com/dataseWts/apollo2506/eurosat-dataset
    """
    def __init__(self, name='apollo2506/eurosat-dataset', **kwargs):
        super().__init__(name, **kwargs)

        self.root_dir = os.path.join(self.root_dir, 'EuroSAT')
        self.train_data = EuroSATSplit(root_dir = self.root_dir, split='train', **kwargs)
        self.val_data = EuroSATSplit(root_dir = self.root_dir, split='validation', **kwargs)
        self.test_data = EuroSATSplit(root_dir = self.root_dir, split='test', **kwargs)
    
    