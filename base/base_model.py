import torch.nn as nn
import numpy as np
from abc import abstractmethod
from functools import partial

from models.prec_conv import Preconditioned_Conv2d
from models.oni import ONI_Conv2d
from models.activation import SReLU, Regular_ReLU

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F

class sn_conv2d(nn.Module):
    """
    Convolution layer encoding spectral normalization
    """
    def __init__(
        self,
        in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=False, padding_mode='zeros'):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv = nn.utils.spectral_norm(self.conv)
        print(self.conv.weight_u.size())
    
    def forward(self, input):
        return self.conv(input)

class Identity(nn.Module):
    """
    Return the input itself
    """

    def __init__(self, out_channel, affine = False):
        super().__init__()

    def forward(self, x):
        return x
    
class BaseModel(nn.Module):
    """
    Base class for all models
    """


    def __init__(self, norm_layer_type, conv_layer_type, linear_layer_type, activation_layer_type):
        super().__init__()
        
        self.norm_layer_type = norm_layer_type
        self.conv_layer_type = conv_layer_type
        self.linear_layer_type = linear_layer_type
        self.activation_layer_type = activation_layer_type

        self.get_norm()
        self.get_conv()
        self.get_linear()
        self.get_activation()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def get_norm(self):
        if self.norm_layer_type == 'bn':
            self.norm_layer = nn.BatchNorm2d
        elif self.norm_layer_type == 'sn':
            self.norm_layer = SwitchNorm2d
        elif self.norm_layer_type == "no":
            # For the case of prec_conv layer
            self.norm_layer = Identity
        else:
            raise NotImplementedError

    def get_conv(self):
        if self.conv_layer_type == 'conv':
            # Since the default setting in most models for bias is False
#             self.conv_layer = partial(nn.Conv2d, bias = False) 
            self.conv_layer = nn.Conv2d
        elif self.conv_layer_type == "prec_conv":
            self.conv_layer = Preconditioned_Conv2d
        elif self.conv_layer_type == "sn_conv":
            # self.conv_layer = partial(sn_conv2d, bias = False)
            self.conv_layer = sn_conv2d
        elif self.conv_layer_type == "oni_conv":
            self.conv_layer = ONI_Conv2d
        else:
            raise NotImplementedError


    def get_linear(self):
        if self.linear_layer_type == 'linear':
            self.linear_layer = nn.Linear
        else:
            raise NotImplementedError

    def get_activation(self):
        if self.activation_layer_type == 'relu':
            self.activation_layer = Regular_ReLU

        elif self.activation_layer_type == 'srelu':
            self.activation_layer = SReLU
        else:
            raise NotImplementedError
        
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
