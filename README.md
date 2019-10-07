## What is it:

Simple Variational Auto Encoder to recreate the input across a Linearizer. Intended to use for Unsupervised clustering of images.

## What does it look like:

    Has a CNN with roughly this architecture:

    `Convolution-->FullyConnected-->DeConvolution`

        The loss function is composed of a pixel to pixel difference in reconstruction (BCE), plus a divergence component (KL-Divergence)

## What does it work on:

    I have a data_loader (misnomer, should be data_set) for CIFAR 10, will actually work with pytorch's CIFAR Dataset also.

## Does it work:

    Kinda!
