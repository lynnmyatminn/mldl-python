# Chapter 2 - The mathematical building blocks of neural networks

## A first look at a neural network
MNIST (Modified National Institue of Standards and Technology)

## Data representations for neural networks
A structure that stores numbers in multiple dimensions.
Most of the data entering into a neural network is stored and processed as tensors.
Tensor - container for data (usually storing numbers). Most machine learning systems use tensors as a basic data structure. Tensors are a generalized version of matrices, to arbitrary number of dimensions. And a dimension is also called *axis* in the tensor context.

- 4 keywords to understand Tensor
1. Axis (A dimension)
2. Rank (Number of axes)
3. Shape (How many items are on each axis?)
4. dtype (Data types, e.g. float32, uint8, bool, ...)

**0D tensor: Scalar**
A scalar is a single number(1 digit)

**1D tensor: Vector**

**2D tensor: Matrix**

**3D tensor: Collection of matrix**
for example, a grayscale iamge dataset:
60000 images x 28 height x 28 width
The shape is -
(60000,28,28)
In the MNIST example in the book, train_images is 3D tensor with a shape of (60000,28,28). The dtype is uint8, and each image is 28x28 grayscale pixel values 0 to 255.
This means-
60000 = number of images/samples
28= image height
28= image width

**4D tensor: Image batch**
In deep learning, image data is trained in batches.
A color image has-
height x width x color_channels
In case of RGB images, there are 3 channels.
Red, Green, Blue
In case of a batch of images-
(samples, height, width, channels)
For example-
(128, 256, 256, 3)
That is-
128 color images
each image=256x256
3 color channels=RGB
Image tensors are used as a batch of rank-3 images and rank-4 images by convention. Keras supports both channels-last (samples, height, width, color_depth) and channels-first (samples, color_depth, height, width).

**5D tensor: Video data**
Video data is a sequence of image frames.
A video-
frames x height x width x channels
A video batch-
(samples, frames, height, width, channels)
For example-
(4, 240, 144, 256, 3)
That means-
4 videos
each video has 240 frames
each frame is 144 x 256
RGB channels = 3

### What is a batch?
Neural networks do not train the entire dataset at once. They train in small chunks. This chunk is called a batch.
For example, in MNIST-
batch = train_images[:128]
the first 128 images are taken as a batch.
In simple terms-
Dataset = entire book
Batch = page/group read at a time

### Why use batch?
It saves memory, faster training, weight updates can be done step by step.

### Tensor slicing
Slicing is the process of taking a portion of data from a Tensor.
For example-
my_slice = train_images[10:100]
It takes image numbers 10 to 99. The shape is- (90, 28, 28)
Another example-
my_slice = train_images[:, 14:, 14:]
It takes the bottom-right 14x14 pixel area of all images. Tensor slicing is explained as data selection along each axis.

### Loss function/ Cost function/ Objective function

## The gears of neural networks: Tensor operations
Actual math operations inside neural networks:
Element-wise operations
broadcasting
dot products
reshaping
geometric interpretation