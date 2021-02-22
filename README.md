# ConvNorm
This repository is the official implementation of Convolutional Normalization: Improving Robustness and Training for Deep Neural Networks

We introduce a simple and effi-cient “convolutional normalization” method thatcan fully exploit the convolutional structure in theFourier domain and serve as a simple plug-and-play module to be conveniently incorporated intoany ConvNets. We show that convolutional normalization can reduce the layerwise spectral norm of the weight matrices and hence improve the Lipschitzness of the network, leading to easier training and improved robustness for deep ConvNets. Applied to classification under noise corruptions and generative adversarial network (GAN), we show that convolutional normalization improves the robustness of common ConvNets such as ResNet and the performance of GAN.

<p float="left" align="center">
<img src="ConvNorm_concept.png" width="800" /> 
<figcaption align="center">
These graphs show the results of training a ResNet-34 with a traditional cross entropy loss (top row) and our proposed method (bottom row) to perform classification on the CIFAR-10 dataset where 40% of the labels are flipped at random. The left column shows the fraction of examples with clean labels that are predicted correctly (green) and incorrectly (blue). The right column shows the fraction of examples with wrong labels that are predicted correctly (green), memorized (the prediction equals the wrong label, shown in red), and incorrectly predicted as neither the true nor the labeled class (blue). The model trained with cross entropy begins by learning to predict the true labels, even for many of the examples with wrong labels, but eventually memorizes the wrong labels. Our proposed method based on early-learning regularization prevents memorization, allowing the model to continue learning on the examples with clean labels to attain high accuracy on examples with both clean and wrong labels.
</figcaption>
</p>
