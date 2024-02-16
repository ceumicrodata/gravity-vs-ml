#################################
# Class for DeepGravity
##################################

import torch
import random
import numpy as np
from typing import Any, Dict, List

class DeepGravityReluOutput(torch.nn.Module):
    """Class for a pytorch neural network module"""
    def __init__(self,
                 dim_input: Any,
                 dim_hidden: Any,
                 dropout_p: int=0.35,
                 num_layers: int=5,
                 device=torch.device("cpu")
                ) -> None:
        """
        Initialize a pytorch model

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

        # random seeds
        random.seed(1)     # python random generator
        np.random.seed(1)  # numpy random generator

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch_device = torch.device("cpu")

        torch.use_deterministic_algorithms(True)

        for layer in range(1, self.num_layers):
            if layer == 1:
                setattr(self, f'linear{layer}', torch.nn.Linear(dim_input, dim_hidden))
                setattr(self, f'relu{layer}', torch.nn.LeakyReLU())
                setattr(self, f'dropout{layer}', torch.nn.Dropout(p))
            else:
                setattr(self, f'linear{layer}', torch.nn.Linear(dim_hidden, dim_hidden))
                setattr(self, f'relu{layer}', torch.nn.LeakyReLU())
                setattr(self, f'dropout{layer}', torch.nn.Dropout(p))

        setattr(self, 'linear_last', torch.nn.Linear(dim_hidden, 1))

        self.relu_out = torch.nn.ReLU()

    def forward(self,
                vX: torch.Tensor) -> torch.Tensor:

        for layer in range(1, self.num_layers):
            if layer == 1:
                lin = getattr(self, f'linear{layer}')(vX)
                relu = getattr(self, f'relu{layer}')(lin)
                drop = getattr(self, f'dropout{layer}')(relu)
            else:
                lin = getattr(self, f'linear{layer}')(drop)
                relu = getattr(self, f'relu{layer}')(lin)
                drop = getattr(self, f'dropout{layer}')(relu)

        lin = self.linear_last(drop)
        
        out = self.relu_out(lin)

        return out

    def loss(self, output, target):
        lsm = torch.nn.LogSoftmax(dim=1)
        sm = torch.nn.Softmax(dim=1)

        output = torch.squeeze(output, dim=-1)
        target = torch.squeeze(target, dim=-1)

        return -(sm(target) * lsm(output)).sum()


class DeepGravityLinOutput(torch.nn.Module):
    """Class for a pytorch neural network module"""
    def __init__(self,
                 dim_input: Any,
                 dim_hidden: Any,
                 dropout_p: int=0.35,
                 num_layers: int=5,
                 device=torch.device("cpu")
                ) -> None:
        """
        Initialize a pytorch model

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

        # random seeds
        random.seed(1)     # python random generator
        np.random.seed(1)  # numpy random generator

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch_device = torch.device("cpu")

        torch.use_deterministic_algorithms(True)

        for layer in range(1, self.num_layers):
            if layer == 1:
                setattr(self, f'linear{layer}', torch.nn.Linear(dim_input, dim_hidden))
                setattr(self, f'relu{layer}', torch.nn.LeakyReLU())
                setattr(self, f'dropout{layer}', torch.nn.Dropout(p))
            else:
                setattr(self, f'linear{layer}', torch.nn.Linear(dim_hidden, dim_hidden))
                setattr(self, f'relu{layer}', torch.nn.LeakyReLU())
                setattr(self, f'dropout{layer}', torch.nn.Dropout(p))

        setattr(self, 'linear_last', torch.nn.Linear(dim_hidden, 1))

    def forward(self,
                vX: torch.Tensor) -> torch.Tensor:

        for layer in range(1, self.num_layers):
            if layer == 1:
                lin = getattr(self, f'linear{layer}')(vX)
                relu = getattr(self, f'relu{layer}')(lin)
                drop = getattr(self, f'dropout{layer}')(relu)
            else:
                lin = getattr(self, f'linear{layer}')(drop)
                relu = getattr(self, f'relu{layer}')(lin)
                drop = getattr(self, f'dropout{layer}')(relu)

        out = self.linear_last(drop)

        return out

    def loss(self, output, target):
        lsm = torch.nn.LogSoftmax(dim=1)
        sm = torch.nn.Softmax(dim=1)

        output = torch.squeeze(output, dim=-1)
        target = torch.squeeze(target, dim=-1)

        return -(sm(target) * lsm(output)).sum()

