# Final Project Datasets

## Iris Dataset

The Iris Dataset consists of 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are not linearly separable from each other.

## MNIST Dataset

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. These images are 28x28 pixels and anti-aliased (greyscaled). A sample MNIST reader in Python can be found here (https://www.kaggle.com/code/hojjatk/read-mnist-dataset).

## CIFAR-10 Dataset

For the CIFAR-10 dataset (binary version, see https://www.cs.toronto.edu/~kriz/cifar.html for more info), The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:
```
<1 x label><3072 x pixel>
...
<1 x label><3072 x pixel>
```
In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.

There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i.



