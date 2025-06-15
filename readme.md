# MLSP Project
This project is developed for the course machine Learning in Signal Processing. Goal of this project was to apply best habits for machine learning in *patch classification* on the EuroSAT dataset.

Goal was to implement a tiny ResNet-18 (<1.5 Mio. Parameters) and apply it to the standard usecase of patch classification regarding land coverage classification.

This project aims to be an easily extendable framework for machine-learning applications. By this, it may get a bit more complicated but serves the possibility to be easily extended to other usecases and architectures.

The idea of this framework style came from mmdetection3d, which I used a lot in my job. It adapts this style in a lightweight manner, trying to be understandable in every way. 

The framework relies on a configuration-based setup - meaning that it is possible, to build and run the entire pipeline only using a configuration dict. This is possible due to the Registries and the Builder.


## Basic Functionalities

### Overview
In the model design process, basic modules build the core building blocks. Important module catagories are: 

|Category|path|Usecase|
|---|---|---|
|`models` | `src.modules` + `subcategory`| Collection of `nn.Modules` building the final trainable model |
|`datasets` | `src.datasets` | The dataset classes to read the dataset files |
|`runner` | `src.runner` | The actual runner, running the train, val or test loops |
|`evaluators` | `src.evaluator` | The evaluator pipeline


### Model building

The idea is that we always want to build a top-level model, found in `src.modules.models`. Those are usually models interconnecting preprocessors, backbones and heads in a usable manner.

```python
import torch.nn as nn

from src.builder import build_module
from src.registry import MODULES

@MODULES.register_module
class FooModel(nn.Module):
    def __init__(self, backbone:nn.Module|dict, head: nn.Module|dict, **kwargs):
        self.backbone = MODULES.build_module(backbone)
        self.head = MODULES.build_module(head)
        ...
    def forward(self, x, **kwargs):
        ...
        x = self.backbone(x)
        ...
        x = self.head(x)
        ...
```

Those modules can either be build directly (the classical style)
```python
from src.modules.models import FooModel
from src.modules.backbone import FooBackbone
from drc.modules.head import FooHead

backbone = FooBackbone()
head = FooHead()
model = FooModel(backbone = backbone, head = head)
```

or by simply providing configuration dicts

```python
from src.registry import MODULES

backbone = {
    'type': 'FooBackbone',
    'in_feat': 3,
    ...
}

head = {
    'type': 'FooHead',
    'in_feat': 512,
    'out_feat':10
    ...
}

model = {
    'type': 'FooModel', 
    'backbone': backbone, 
    'head': head
    }
model = MODULES.build_module(model)
```

This way, the architecture gains a high level of flexibility. 


### Run Model


To run the programmed model the user has to provide specific keys in the config dict or instantiate the specific submodules him- or herself.

The **Runner** (src.runner) allows the user build all necessary modules to execute either train, validation or test loops. It takes care of logging and validation steps inside the train loops and also executes the Evaluator.

```pyhon
def __init__(self, model: nn.Module | dict,
                 dataloader_cfg: dict,
                 dataset: nn.Module | dict,
                 optim: nn.Module | dict,
                 work_dir: str = None,
                 device: str = 'cpu', 
                 seed: int = None,
                 **kwargs):
        """
        Initializes the Runner with model, data, and optimizer configurations.
        
        Args:
            model (nn.Module | dict): Model configuration or instance.
            dataloader_cfg (dict): Configuration for DataLoader.
            dataset (nn.Module | dict): Dataset configuration or instance.
            optim (nn.Module | dict): Optimizer configuration or instance.
            work_dir (str, optional): Directory to save model and logs. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
```

The main entrypoint in the model execution is 
```python
def run(self,
            mode: str = 'train',
            val_interval: int= 10,
            log_interval: int= 10,
            epochs: int = 100,
            start_epoch: int = 1):
        """
        Main entry point for running the model.
        Args:
            mode (str): 'train', 'validation', or 'test'.
            val_interval (int): Validation interval in epochs.
            log_interval (int): Logging interval in batches.
            epochs (int): Total number of epochs to train.
            start_epoch (int): Starting epoch for training.
        Returns:
            dict: History of training/validation metrics.
        """
```


# Specific classes


### Models `src.modules.models`

| Module                             | Path                         | Description                                                                 |
|------------------------------------|------------------------------|-----------------------------------------------------------------------------|
| **EuroSATModel**                  | `src.model`                  | ResNet backbone with a Feed-Forward Network (FFN) for classification.      |


### Datasets `src.datasets`
| Class        | Description                                                        |
|--------------|--------------------------------------------------------------------|
| `EuroSAT`    | EuroSAT dataset loader with `Split` and `DatasetContainer` support. |
| `ImageNet`   | Tiny ImageNet loader with `Split` and `DatasetContainer` support.   |


### Modules `src.modules`
#### Backbones `src.modules.backbones`
| Class     | Description                        |
|-----------|------------------------------------|
| `ResNet`  | Standard ResNet used as backbone.  |
#### Heads `src.modules.heads`
| Class   | Description                           |
|---------|---------------------------------------|
| `FFN`   | Basic Feed-Forward Network head.      |

### Runner `src.runner`
| Class     | Description                                      |
|-----------|--------------------------------------------------|
| `Runner`  | Main loop: handles training, validation, testing.|

### Evaluators `src.evaluators`
| Class           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `BaseEvaluator` | Computes mAP, F1 score, and confusion matrix.|
















