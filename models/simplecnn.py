from base import BaseModel
import torch
import torch.nn as nn


KERNEL_SIZE = 3
class CNN(BaseModel):
    def __init__(self, num_classes=10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
        super().__init__(norm_layer_type, conv_layer_type, linear_layer_type, activation_layer_type)

        cfg = [8, 8, 'M', 16, 16, 'M', 32, 32, 'M']
        self.features = self._make_layers(cfg)
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
                layers += [self.conv_layer(in_channels = in_channels, out_channels = x, kernel_size=KERNEL_SIZE, padding=int(KERNEL_SIZE/2)),
                           self.norm_layer(x, affine = True),
                           self.activation_layer(x, inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)
