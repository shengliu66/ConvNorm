from base import BaseModel
import torch
import torch.nn as nn

class Net_circular_CNN(BaseModel):
    def __init__(self, num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
        super().__init__(norm_layer_type, conv_layer_type, linear_layer_type, activation_layer_type)
        
        self.conv1 = self.conv_layer(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = self.norm_layer(6, affine = True)

        self.conv2 = self.conv_layer(6, 16, 3, bias=False)
        self.norm2 = self.norm_layer(16, affine = True)

        self.conv3 = self.conv_layer(16, 20, 3, bias=False)
        self.norm3 = self.norm_layer(20, affine = True)

        self.conv4 = self.conv_layer(20, 32, 3, bias=False)
        self.norm4 = self.norm_layer(32, affine = True)

        self.fc = nn.Linear(32,num_classes)
        
    def forward(self, x):
        x = ((self.pool(F.relu(self.norm1(self.conv1(x))))))
        x = ((self.pool(F.relu(self.norm2(self.conv2(x))))))
        x = ((F.relu(self.norm3(self.conv3(x)))))
        x = ((self.pool(F.relu(self.norm4(self.conv4(x))))))
        x = x.view(-1, 32)
        x = self.fc(x)
        return x
