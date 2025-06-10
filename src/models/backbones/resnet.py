from .utils import BasicBlock
from torch import nn

from src.models.backbones.utils import BasicBlock

class ResNet(nn.Module):

    def __init__(self, idims, odims, arch, base_dims=32):
        """
        ResNet model for image classification.

        Args:
            in_feat (int): Number of input features (channels).
            out_feat (int): Number of output features (classes).
            arch (tuple): Architecture of the ResNet model, defining the number of layers in each stage.
            hidden_feat (int): Number of hidden features in the first layer.
            pretrained (bool): If True, load pretrained weights.
        """
        super(ResNet, self).__init__()

        self.idims = idims
        self.base_dims = base_dims
        self.out_feat = odims
        self.arch = arch

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Conv2d(self.idims, self.base_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.base_dims),
            nn.ReLU(inplace=True),
        ))
        
        ifeat = self.base_dims
        for stage in self.arch:
            self.layers.extend(self._make_layer(ifeat, ifeat * 2, stage))
            ifeat *= 2

        self.layers.append(nn.Conv2d(in_channels=ifeat, out_channels=self.out_feat, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        Forward pass of the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_feat, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_fear).
        """
        for layer in self.layers:
            x = layer(x)

        return x

    def _make_layer(self, in_channels, out_channels, layers=1, **kwargs):
        """
        Create a stage of the ResNet architecture.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            layers (int): Number of layers in this stage.
        """

        layers_list = nn.ModuleList()

        layers_list.append(BasicBlock(in_channels, out_channels, stride=2, **kwargs))

        for i in range(layers-1):
            layers_list.append(BasicBlock(out_channels, out_channels, **kwargs))

        return layers_list
