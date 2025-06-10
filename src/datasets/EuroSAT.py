from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms


class EuroSATDataset(Dataset):
    """EuroSAT Dataset for satellite image classification.
    
    Args:
        root_dir (str): Directory with all the images.
        split (str): Split of the dataset to use ('train', 'val', 'test').
                     The split should correspond to a CSV file named '{split}.csv' in the root directory.
                     The CSV file should have two columns: 'Filename' and 'Label'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.split = split

        split_file = os.path.join(root_dir, f"{split}.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} does not exist.")
        df = pd.read_csv(split_file)

        self.image_paths = df['Filename'].tolist()
        self.labels = df['Label'].tolist()  
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        label = torch.tensor(self.labels[idx])

        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        
        return {'image': image, 'label': label}