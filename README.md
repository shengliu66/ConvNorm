# ConvNorm
This repository is the official implementation of [Convolutional Normalization: Improving Robustness and Training for Deep Neural Networks](https://arxiv.org/abs/2103.00673).

We introduce a simple and efficient “convolutional normalization” method thatcan fully exploit the convolutional structure in theFourier domain and serve as a simple plug-and-play module to be conveniently incorporated intoany ConvNets. We show that convolutional normalization can reduce the layerwise spectral norm of the weight matrices and hence improve the Lipschitzness of the network, leading to easier training and improved robustness for deep ConvNets. Applied to classification under noise corruptions and generative adversarial network (GAN), we show that convolutional normalization improves the robustness of common ConvNets such as ResNet and the performance of GAN.

<p float="left" align="center">
<img src="ConvNorm_concept.png" width="800" /> 
<figcaption align="center">
These graphs show the comparison between BatchNorm and ConvNorm on activations of k=1,...,C channels. BatchNorm subtracts and multiplies the activations of each channel by computed scalars: mean µ and variance σ, before a per-channel affine transform parameterized by learned parameters β and γ; ConvNorm performs per-channel convolution with precomputed kernel v to normalize the spectrum of the weight matrix for the convolution layer, following with a channel-wise convolution with learned kernel r as the affine transform..
</figcaption>
</p>

## Environment
- python 3.9.0
- torch 1.6.0
- torchvision 0.7.0
- sklearn 0.23.2
- comet_ml 3.2.11
- numpy 1.19.2
- matplotlib 3.3.2

The versions of the dependencies listed are not strict, some earlier versions of the dependencies should also work.

## General
We use config files to control the parameters of training and validation. An example of standard training config could be found in `config_cifar10_standard.json` and an example of robustly training could be found in `config_robust_train.json`. We will mention the important parameters for each set up below.

In order to use our proposed Convolutional Normalization (ConvNorm), you can simply replace your convolutional layer and normalization layer by this [module](./models/prec_conv.py). In the module, the parameter **affine** controls whether to use the affine transform of our method, and the parameter **bn** controls whether to use BatchNorm after ConvNorm.

## Training
### Data
- Please download the data before running the code, add path to the downloaded data to `data_loader.args.data_dir` in the corresponding config file.
### Training
Change `arch.args.conv_layer_type` for different methods and set `arch.args.norm_layer_type` accordingly. For example, to run standard convolution with BatchNorm, set `arch.args.conv_layer_type` to be "conv" and `arch.args.norm_layer_type` to be "bn"; to run our our method, set `arch.args.conv_layer_type` to be "prec_conv" and `arch.args.norm_layer_type` to be "no" (since we include the BatchNorm in the implementation of our method) in the config file.

- Code for standard training with ConvNorm is in the following file: `train.py`. For reproducing our label noise experiments, set `trainer.sym` to True and denote the desired label noise percentage by changing `trainer.percent`. For reproducing the data scarcity experiments, change `trainer.subset_percent` to the desired subset percentage.
- Code for rubust training of CIFAR10 is in the following file: `train_robust.py`. A corresponding config file is `config_robust_train.json`. Note that in the config, `trainer.adv_repeats` means the times for repeating training for each minibatch; `trainer.fgsm_step` denotes the step amount of one FGSM iteration; `trainer.adv_clip_eps` is the $l_{inf}$ norm bound of the perturbation; `pgd_attack.K` is the total steps for PGD attack; `pgd_attack.step` is the attack amount of one PGD step; `trainer.OCNN` denotes whether to use OCNN method or not (different than other methods which should be set in `arch` since OCNN is a regularization method enforced directly to the loss) and `trainer.lamb_ocnn` is the correpsonding penalty constraint constant of OCNN.

Example usage:
~~~python
$ python train.py -c config_cifar10_standard.json
$ python train_robust.py -c config_robust_train.json
~~~

## Validation
`validate_pgd.py` and `fgsm.py` are the validation codes for measuing the performances of the robustly trained models, they should be used with the config file `config_robust_train.json` as well.

Example usage:
~~~python
$ python validate_pgd.py -c config_robust_train.json -s <model checkpoint directory> --seed <random seed>
$ python fgsm.py -c config_robust_train.json -s <model checkpoint directory> --seed <random seed>
~~~

## Other
`check_result.ipynb` contains some example usage of the validation codes and the corresponding results from our experiments.


## Reference
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2103.00673).
```
@article{liu2021convolutional,
      title={Convolutional Normalization: Improving Deep Convolutional Network Robustness and Training}, 
      author={Sheng Liu and Xiao Li and Yuexiang Zhai and Chong You and Zhihui Zhu and Carlos Fernandez-Granda and Qing Qu},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
}
```

## Contact
Please contact shengliu@nyu.edu or xl998@nyu.edu if you have any question on the codes.
