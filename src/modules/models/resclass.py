from torch import nn
import torch

from src.modules.backbones.resnet import ResNet
from src.modules.heads import FFN

class EuroSATModel(nn.Module):
    def __init__(self, backbone_cfg, head_cfg, **kwargs):
        super(EuroSATModel, self).__init__()
        self.backbone = backbone_cfg['type'](**backbone_cfg)
        self.head = head_cfg['type'](**head_cfg)
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
    
    def describe(self):
        return {
            'backbone': self.backbone.describe(),
            'head': self.head.describe(),
            'criterion': self.criterion.__class__.__name__
        }
    def print_param_summary(self):
        """
        Prints a summary of the model parameters, including trainable and non-trainable parameters.
        """
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        num_non_trainable = num_total - num_trainable

        # Prepare labels & values
        rows = [
            ("Trainable params",      num_trainable),
            ("Non-trainable params",  num_non_trainable),
            ("Total params",          num_total),
            ("- Backbone params",       backbone_params),
            ("- Head params",           head_params),
            
        ]

        # Compute column widths
        label_width = max(len(label) for label, _ in rows)
        value_width = max(len(f"{value:,}") for _, value in rows)
        table_width = label_width + value_width + 5  # padding for " | "

        # Print header
        print("=" * table_width)
        print("Model Parameters Summary".center(table_width))
        print("-" * table_width)

        # Print rows
        for label, value in rows:
            print(f"{label:<{label_width}} | {value:>{value_width},}")

        # Print footer
        print("=" * table_width)