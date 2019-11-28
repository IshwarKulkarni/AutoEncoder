## What is it:

Simple Variational Auto Encoder to recreate the input across a Linearizer. 
Intended to use for Unsupervised Clustering of images using the latent space representation of images. 

## What does it look like:

Has a ResNet like CNN with roughly this architecture:

 `Convolution --> FullyConnected (op: Latent Space) -->DeConvolution`

The loss function is composed of a pixel to pixel difference in reconstruction (BCE),
plus a divergence component (KL-Divergence) : vanilla  VAE losses. 

## What does it work on:

I have a data_loader (misnomer, should be data_set) for CIFAR 10, will actually work with pytorch's CIFAR Dataset also.

## Does it work:

Kinda!
Does a good job on MNIST data set
CIFAR not so much
Testing some real world data  
