from .vgg import VGG
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .simplecnn import CNN
from .simplecnn2 import Net_circular_CNN

def vgg16(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return VGG('VGG16',num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)

def vgg19(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return VGG('VGG19',num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)

# Resnet
def resnet18(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet18(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type)

def resnet34(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet34(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type)

def resnet50(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet50(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type)

def resnet101(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet101(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                     activation_layer_type = activation_layer_type)

def resnet152(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet152(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                     activation_layer_type = activation_layer_type)

# Simple model
def simpleCNN(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)

def simpleCNN2(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return Net_circular_CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)
