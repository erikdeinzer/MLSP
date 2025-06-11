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

This way, the architecture gains high level of flexibility. 















