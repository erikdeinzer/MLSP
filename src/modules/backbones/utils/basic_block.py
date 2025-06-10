import torch

from torch import nn



import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=None, **kwargs):
        super(BasicBlock, self).__init__()
        # First conv: 3×3, possibly with stride>1 (for spatial downsampling)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # Second conv: 3×3, always stride=1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2   = nn.BatchNorm2d(out_channels)

        # A downsample module for the shortcut
        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        # Optional dropout *after* the second ReLU
        self.dropout = nn.Dropout(dropout_rate) if (dropout_rate is not None) else None

    def forward(self, x):
        identity = x

        # First conv → BN → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv → BN
        out = self.conv2(out)
        out = self.bn2(out)

        # If downsample is provided, apply it to the residual branch
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out = out + identity
        out = self.relu(out)

        # Apply dropout (if any) after the ReLU
        if self.dropout is not None:
            out = self.dropout(out)

        return out
