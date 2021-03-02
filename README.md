# ConvNorm
This repository is the official implementation of Convolutional Normalization: Improving Robustness and Training for Deep Neural Networks

We introduce a simple and efficient “convolutional normalization” method thatcan fully exploit the convolutional structure in theFourier domain and serve as a simple plug-and-play module to be conveniently incorporated intoany ConvNets. We show that convolutional normalization can reduce the layerwise spectral norm of the weight matrices and hence improve the Lipschitzness of the network, leading to easier training and improved robustness for deep ConvNets. Applied to classification under noise corruptions and generative adversarial network (GAN), we show that convolutional normalization improves the robustness of common ConvNets such as ResNet and the performance of GAN.

<p float="left" align="center">
<img src="ConvNorm_concept.png" width="800" /> 
<figcaption align="center">
These graphs show the comparison between BatchNorm and ConvNorm on activations of k=1,...,C channels. BatchNorm subtracts and multiplies the activations of each channel by computed scalars: mean µ and variance σ, before a per-channel affine transform parameterized by learned parameters β and γ; ConvNorm performs per-channel convolution with precomputed kernel v to normalize the spectrum of the weight matrix for the convolution layer, following with a channel-wise convolution with learned kernel r as the affine transform..
</figcaption>
</p>

## Training
### Data
- Please download the data before running the code, add path to the downloaded data to `data_loader.args.data_dir` in the corresponding config file.
### Training
- Code for training CIFAR100 with ConvNorm is in the following file: [`train.py`](./train.py).
```
usage: train.py [-c] [-r] [-d] [--lr learning_rate] [--bs batch_size] [--conv conv_layer] [--norm norm_layer] [--seed seed]
                               [--name exp_name] 

  arguments:
    -c, --config                  config file path (default: None)
    -r, --resume                  path to latest checkpoint (default: None)
    -d, --device                  indices of GPUs to enable (default: all)     
  
  options:
    --lr learning_rate            learning rate (default value is the value in the config file)
    --bs batch_size               batch size (default value is the value in the config file)
    --conv conv_layer             type of convolution layer to use (for ours: prec_conv, for default conv: conv)
    --norm norm_layer             type of normalization layer to use (for ours: no, for default BatchNorm: bn)
    --seed seed                   which random seed to set
    --name exp_name               experiment name
```
Configuration file is **required** to run the training, other types of option could be found in the following file: [`train.py`](./train.py).
