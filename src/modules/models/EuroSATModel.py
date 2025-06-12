from torch import nn
import torch

from src.modules.models.base_model import BaseModel

from src.build.registry import MODULES

@MODULES.register_module()
class EuroSATModel(BaseModel):
    def __init__(self, backbone_cfg, head_cfg, **kwargs):
        super(EuroSATModel, self).__init__(backbone_cfg, head_cfg, **kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.flatten = nn.Flatten()
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
    
    def loss(self, input, target):
        logits = self.forward(input)
        loss = self.criterion(logits, target)
        return loss
    
    def forward_train(self, x, target):
        img = x['image']
        logits = self.forward(img)
        loss = self.criterion(logits, target)
        return logits, loss
    
    def forward_test(self, x):
        img = x['image']
        logits = self.forward(img)
        return logits
    
    def predict(self, x):
        logits = self.forward_test(x)
        return torch.argmax(logits, dim=1)