import torchvision.transforms as transforms

from src.datasets import EuroSATDataset
from src.modules.backbones.resnet import ResNet
from src.modules.heads import FFN
from src.modules.models import EuroSATModel
from src.runner import Runner
import os

import kagglehub, os

# Download latest version
path = kagglehub.dataset_download("apollo2506/eurosat-dataset")

print("Path to dataset files:", path)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

loading_cfg = {
    'batch_size': 1024,
    'num_workers': 4,
}

data_cfg = {
    'root_dir': os.path.join(path, 'EuroSAT'),
    'transform': transform,
}

optim_cfg = {
    'lr': 0.001,
    'weight_decay': 1e-4,
}


model_cfg = {
    'type': EuroSATModel,
    'backbone_cfg': {
        'type': ResNet,
        'idims': 3,
        'odims': 64,
        'base_dims': 12,
        'arch': [2,2,2,2],
        'dropout': 0.2,
    },
    'head_cfg': {
        'type': FFN,
        'idims': 64,
        'odims': 10, # Number of classes in EuroSAT dataset
        'hidden_dims': 1024,
        'nlayers':6,
        'dropout': 0.2,
    }
}

runner = Runner(model_cfg=model_cfg, loading_cfg=loading_cfg, data_cfg=data_cfg, optim_cfg=optim_cfg, device='cuda:2', work_dir='results/eurosat')

runner.run(mode='train', val_interval=1, log_interval=1, epochs=100, start_epoch=1)