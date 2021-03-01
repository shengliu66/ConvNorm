import torch
import torch.nn as nn
import numpy as np

class Regular_ReLU(nn.Module):
    """Regular ReLU"""

    def __init__(self, nc, inplace=True):
        super(Regular_ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)

class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, inplace=True):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=inplace)
        nn.init.constant_(self.srelu_bias, -1.0)

        setattr(self.srelu_bias, 'srelu_bias', True)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias

