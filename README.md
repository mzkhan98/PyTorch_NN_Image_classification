# Simple to Advanced Neural Networks in PyTorch

In this repository, we present three different neural network architectures using PyTorch: a simple neural network, a slightly complex neural network, and an advanced convolutional neural network (CNN). We demonstrate the implementation, training, and evaluation of each model.

## Simple Neural Network

The simple neural network contains a single linear layer. This model achieved an accuracy of 92.2% on the test dataset.

**Model architecture:**

- Linear layer (input_size, num_classes)

## Slightly Complex Neural Network

The slightly complex neural network is a two-layer feedforward network with a hidden layer and a ReLU activation function. This model achieved an accuracy of 97.49% on the test dataset.

**Model architecture:**

- Linear layer (input_size, hidden_size)
- ReLU activation
- Linear layer (hidden_size, num_classes)

## Advanced Convolutional Neural Network (CNN)

The advanced CNN consists of two convolutional layers followed by two fully connected layers. This model achieved an accuracy of 98.24% on the test dataset..

**Model architecture:**

- Convolutional layer (1, 16, kernel_size=3, stride=1, padding=1)
- ReLU activation
- Max pooling (kernel_size=2, stride=2)
- Convolutional layer (16, 32, kernel_size=3, stride=1, padding=1)
- ReLU activation
- Max pooling (kernel_size=2, stride=2)
- Fully connected layer (32 * 7 * 7, 128)
- ReLU activation
- Dropout (p=0.5)
- Fully connected layer (128, num_classes)

## Training and Evaluation

For each model, we used the following setup:

- Loss function: Cross-entropy loss
- Optimizer: Adam with a specified learning rate
- Training loop: Forward pass, loss computation, backward pass, and optimizer step
- Evaluation: Test accuracy computation using the trained model

## Getting Started

To run the code, you'll need to have PyTorch installed on your machine. You can find the installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

Once you have PyTorch installed, clone this repository and run the Python files corresponding to the models.

## Learn More

If you are new to neural networks and PyTorch, we recommend checking out the following resources:

- [PyTorch official documentation](https://pytorch.org/docs/stable/index.html)
- [Deep Learning with PyTorch: A 60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Convolutional Neural Networks (CNNs) Explained](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).