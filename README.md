# ImageNet Classification with ResNet-50

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/AWS-g6dn.8xlarge-FF9900?logo=amazon-aws)](https://aws.amazon.com/)
[![Training Time](https://img.shields.io/badge/Training%20Time-6%20hours-green.svg)]()
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Deployed-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository implements an ImageNet classification pipeline using a ResNet-50 model built from scratch with PyTorch. The project includes data preparation, training, evaluation, and utility scripts to streamline experimentation.

# ImageNet dataset

The ImageNet-1K dataset is a widely used benchmark in computer vision and deep learning research. It consists of 1,000 classes and approximately 1.2 million training images. The dataset also includes 50,000 validation images and is designed to challenge models with a diverse set of object categories, ranging from animals and plants to everyday objects and scenes.

# AWS setup

- Instance Type: AWS g6dn.8xlarge

- Storage: 500 GB EBS volume for dataset storage

- Training Configuration:

    - Batch size: 256

    - Epochs: 3

    - Training time per epoch: ~2 hours

- Results: Achieved 39.6% Top-1 accuracy and 66.0% Top-5 accuracy after 3 epochs.
  
  Note: We could run only 3 epochs as we were running out of time to exhaust our AWS credits. Can get better accuracy with more number of epochs.

# Model Architecture:

ResNet-50: Built with a total of 50 layers, consisting of initial 7x7 convolution and max pooling, 4 stages of bottleneck residual blocks (3, 4, 6, 3 blocks per stage),
Adaptive average pooling and a fully connected (FC) layer

Key Features:

- Residual Connections: Skip connections for efficient gradient flow.
- Bottleneck Design: Efficient 3-layer blocks (1x1, 3x3, 1x1 convolutions).
- Output: Predicts class probabilities (default: 1000 classes).
- Input Shape: 3-channel images (e.g., RGB) of size 224x224 or larger.
- Output Shape: Probability scores for the number of target classes.

# Model Training on EC2

![Alt text](train2_1.jpg)

# GPU Utilization on EC2
![Alt text](train1_1.jpg)


# Model Deployment:

HuggingFace 
https://huggingface.co/spaces/hotshotdragon/TSAI-Imagenet-Resnet

![image](https://github.com/user-attachments/assets/e4a4dfb4-d8f9-47bf-ae58-9e4f8a1f8333)


# Future Work:
- Training the model for more number of epochs and improving the accuracy

