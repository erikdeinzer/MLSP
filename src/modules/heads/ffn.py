import torch
from torch import nn

from src.build.registry import MODULES

@MODULES.register_module()
class FFN(nn.Module):
    def __init__(self, idims, odims, nlayers, hidden_dims=512, dropout_rate=None, **kwargs):
        """
        Feed-Forward Network (FFN) for classification tasks.

        Args:
            idims (int): Number of input features.
            odims (int): Number of output features (classes).
            hidden_dims (int): Number of hidden dimensions.
            dropout_rate (float): Dropout rate to apply after the first linear layer.
        """
        super(FFN, self).__init__()

        self.idims = idims
        self.odims = odims
        self.hidden_dims = hidden_dims
        
        self.layers = nn.ModuleList()

        self.layers.append(self._make_layer(idims, hidden_dims, dropout_rate=dropout_rate, **kwargs))
        
        for _ in range(nlayers-2):
            self.layers.append(self._make_layer(hidden_dims, hidden_dims, **kwargs))

        self.layers.append(nn.Linear(hidden_dims, odims))
            

    def forward(self, x):
        """
        Forward pass of the FFN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, idims).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, odims).
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _make_layer(self, idims, odims, dropout_rate=None, **kwargs):
        """
        Create a single layer of the FFN.

        Args:
            idims (int): Number of input features.
            odims (int): Number of output features.
            dropout_rate (float): Dropout rate to apply after the first linear layer.
        """
        layers = nn.ModuleList()
        layers.append(nn.Linear(idims, odims))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
        
    def describe(self):
        """
        Get a string description of the FFN model.

        Returns:
            str: Description of the model.
        """
        return f"FFN(idims={self.idims}, odims={self.odims}, hidden_dims={self.hidden_dims})"