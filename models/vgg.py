"""VGG11/13/16/19 in Pytorch."""
from base import BaseModel
import torch
import torch.nn as nn
import argparse
from functools import partial
from parse_config import ConfigParser


KERNEL_SIZE=3
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}


class VGG(BaseModel):
    def __init__(
                self, 
                vgg_name, 
                num_classes=10, 
                norm_layer_type = 'bn',
                conv_layer_type = 'conv2d',
                linear_layer_type = 'linear',
                activation_layer_type = 'relu'):
        super().__init__(norm_layer_type, conv_layer_type, linear_layer_type, activation_layer_type)

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier= self.linear_layer(512,num_classes)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                self.conv_touse = self.conv_layer

                layers += [self.conv_touse(in_channels = in_channels, out_channels = x, kernel_size=KERNEL_SIZE, padding=int(KERNEL_SIZE/2)),
                           self.norm_layer(x, affine = True),
                           self.activation_layer(x, inplace=True)]
                in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)
