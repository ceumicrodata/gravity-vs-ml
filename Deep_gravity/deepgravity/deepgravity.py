#################################
# Class for DeepGravity
##################################

import torch
from typing import Any, Dict, List

class DeepGravity(torch.nn.Module):
    """Class for a pytorch neural network module"""
    def __init__(self,
                 dim_input: Any,
                 dim_hidden: Any,
                 dropout_p: int=0.35,
                 num_layers: int=5,
                 device=torch.device("cpu")
                ) -> None:
        """
        Initialize a pytorch model with DCRNN architecture

        Args:
            dim_input:
            dim_hidden:
            dropout_p:
            device:
        """
        super().__init__()
        p = dropout_p
        self.device = device
        self.num_layers = num_layers

        setattr(self, 'linear1', torch.nn.Linear(dim_input, dim_hidden))
        setattr(self, 'relu1', torch.nn.LeakyReLU())
        setattr(self, 'dropout1', torch.nn.Dropout(p))

        for layer in range(2, num_layers):
            setattr(self, f'linear{layer}', torch.nn.Linear(dim_hidden, dim_hidden))
            setattr(self, f'relu{layer}', torch.nn.LeakyReLU())
            setattr(self, f'dropout{layer}', torch.nn.Dropout(p))

        setattr(self, 'linear_last', torch.nn.Linear(dim_hidden, dim_hidden // 2))
        setattr(self, 'relu_last', torch.nn.LeakyReLU())
        setattr(self, 'dropout_last', torch.nn.Dropout(p))

        self.linear_out = torch.nn.Linear(dim_hidden // 2, 1)

    def forward(self,
                vX: torch.Tensor) -> torch.Tensor:

        lin = self.linear1(vX)
        relu = self.relu1(lin)
        drop = self.dropout1(relu)

        for i in range(2, self.num_layers):
            lin = self.linear2(drop)
            relu = self.relu2(lin)
            drop = self.dropout2(relu)

        lin = self.linear_last(drop)
        relu = self.relu_last(lin)
        drop = self.dropout_last(relu)

        out = self.linear_out(drop)

        return out

    def loss(self, output, target):
        lsm = torch.nn.LogSoftmax(dim=1)
        sm = torch.nn.Softmax(dim=1)

        output = torch.squeeze(output, dim=-1)
        target = torch.squeeze(target, dim=-1)

        return -(sm(target) * lsm(output)).sum()
