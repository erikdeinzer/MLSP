import torch
import torch.nn as nn

from src.build.registry import build_module, MODULES

class BaseModel(nn.Module):

    def __init__(self, backbone, head, ckpt=None):
        """
        Base model class that combines a backbone and a head for feature extraction and classification.

        Args:
            backbone (nn.Module): Backbone network for feature extraction.
            head (nn.Module): Head network for classification.
            criterion (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
        """
        super(BaseModel, self).__init__()
        self.backbone = build_module(backbone, registry=MODULES)
        self.head = build_module(head, registry=MODULES)
        self.criterion = None
        if ckpt is not None:
            self.load_checkpoint(**ckpt)

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def describe(self):
        return {
            'backbone': self.backbone.__dict__,
            'head': self.head.__dict__,
            'criterion': self.criterion
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
    
    def save_checkpoint(self, path: str, meta: dict = None):
        """
        Save backbone and head state dictionaries to a single checkpoint file.

        Args:
            path (str): Filesystem path to save the checkpoint.
            meta (dict, optional): Additional metadata to include.
        """
        ckpt = {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict()
        }
        if meta is not None:
            ckpt["meta"] = meta
        torch.save(ckpt, path)

        return path

    def load_checkpoint(self,
                         path: str,
                         load_backbone: bool = True,
                         load_head: bool = True,
                         strict: bool = True,
                         map_location: str = "cpu"):
        """
        Load backbone and/or head weights from a checkpoint file.

        Args:
            path (str): Filesystem path to the checkpoint.
            load_backbone (bool): If True, load backbone weights.
            load_head (bool): If True, load head weights.
            strict (bool): Passed to `load_state_dict` to enforce matching keys.
            map_location (str or torch.device): Device mapping for loading.

        Returns:
            self: The model instance with loaded weights.
        """
        ckpt = torch.load(path, map_location=map_location)
        if load_backbone and "backbone" in ckpt:
            self.backbone.load_state_dict(ckpt["backbone"], strict=strict)
        if load_head and "head" in ckpt:
            self.head.load_state_dict(ckpt["head"], strict=strict)
        return self

 
