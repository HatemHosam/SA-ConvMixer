# SA-ConvMixer

This is the test code of SA-ConvMixer paper: Pixel Shuffling is all you Need: Spatially Aware ConvMixer for Dense Prediction Tasks

Requirements:

Tensorflow = '2.10.0'

opencv = '4.6.0'

numpy = '1.21.5'

matplotlib = '3.5.2'

pytorch = '1.13.0'


Note that CIFAR10 classification code is implemented using Pytorch by modifying original Conv-mixer code in here:

https://github.com/locuslab/convmixer-cifar10

The ImageNet-1k experiment is implemented in PyTorch using timm's framework, we modify the code from the original ConvMixer available at:

https://github.com/locuslab/convmixer

All other codes are originally implemented using Tensorflow.
