# Chapter 3 - Introduction to TensorFlow, PyTorch, JAX and Keras

## A brief history of deep learning frameworks

- 3 Key features of Deep Learning frameworks:
1. Automatic differentiation
2. Run Tensor computation on CPU/GPU/AI chips
3. Distributed computation across multiple GPUs / computers

- XLA: high-performance compiler developed to enable TensorFlow to run on TPUs.
- TensorFlow: released in 2015 by Google
- PyTorch: released in 2016 by Meta
- JAX: alternative way to use autodifferentiation with XLA, google.

## How these frameworks relate to each other

- Low-level frameworks: Tensor manipulation (tensors, tensor operations, backprop.)
1. TensorFlow
2. PyTorch
3. JAX

- High-level frameworks: High-level deep learning concepts (layers, loss functions, optimizer, metrics, training loop that performs mini-batch stochastic gradient descent)
1. Keras

## Introduction to TensorFlow