# Final Project Datasets

## MNIST Dataset

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. These images are 28x28 pixels and anti-aliased (greyscaled). A sample MNIST reader in Python can be found here (https://www.kaggle.com/code/hojjatk/read-mnist-dataset).

## CIFAR-10 Dataset

The CIFAR-10 dataset is a labeled subset of the 80 million tiny images dataset. It consists of the following:
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 32x32 pixel images, RGB colored
- 50000 training images, 10000 test images

For the CIFAR-10 dataset (binary version, see https://www.cs.toronto.edu/~kriz/cifar.html for more info), the binary version contains the files `data_batch_1.bin`, `data_batch_2.bin`, `...`, `data_batch_5.bin`, as well as `test_batch.bin`. Each of these files is formatted as follows:
```
<1 x label><3072 x pixel>
...
<1 x label><3072 x pixel>
```
In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.

There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i.


## STL-10 Dataset

The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. It is inspired by the CIFAR-10 dataset but with some modifications. In particular, each class has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples is provided to learn image models prior to supervised training. The primary challenge is to make use of the unlabeled data (which comes from a similar but different distribution from the labeled data) to build a useful prior. The higher resolution of this dataset (96x96) makes it a challenging benchmark for developing more scalable unsupervised learning methods.

- 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- 96x96 pixel images, RGB colored
- 500 training images (10 pre-defined folds), 800 test images per class
- 100,000 unlabeled images (contains images that may not be within the 10 classes)

For the STL-10 dataset (binary version, see https://cs.stanford.edu/~acoates/stl10/ for more info), the binary version contains the files `train_X.bin`, `train_y.bin`, `test_X.bin`, `test_y.bin`, and `unlabeled_X.bin`. The files are formatted as follows:
```
<1 x label><27648 x pixel>
...
<1 x label><27648 x pixel>
```
In other words, the values are stored as tightly packed arrays of uint8's. The images are stored in column-major order, one channel at a time. That is, the first 96x96 values are the red channel, the next 96x96 are green, and the last are blue. The labels are in the range 1-10.

The file `class_names.txt` is included for reference, with one class name per line. The file `fold_indices.txt` contains the zero-based indices of the exampels to be used for each training fold. The first line contains the indices for the first fold, the second line, the second fold, and so on.

Since the dataset is quite large and cannot be stored in GitHub, please run the following commands (in this directory) if you intend to use the STL-10 dataset:
```
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz &&
mkdir -p STL-10 &&
tar -xzvf stl10_binary.tar.gz -C STL-10
```

