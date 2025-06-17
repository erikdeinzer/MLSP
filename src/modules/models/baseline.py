from torch import nn
import torch

from src.modules.models.base_model import BaseModel

from src.build.registry import MODULES

@MODULES.register_module()
class Baseline(BaseModel):
    def __init__(self, backbone_cfg, **kwargs):
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
        )
        super(Baseline, self).__init__(backbone_cfg, head, **kwargs)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.backbone(x)
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