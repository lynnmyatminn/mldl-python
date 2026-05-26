# import the MNIST dataset from keras.datasets and load the data
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print the number of axes of the tensor
print(train_images.ndim)

# print the training images 'shape' property (60000, 28, 28) -> len=60000 each frame or index is a multidimensional array of 28x28 pixels
print(train_images.shape)

# print the data type of the images
print(train_images.dtype)