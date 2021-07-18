from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .simplecnn import CNN
from .simplecnn2 import Net_circular_CNN

# Resnet
def resnet18(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet18(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type)

# Simple model
def simpleCNN(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)

def simpleCNN2(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return Net_circular_CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)
