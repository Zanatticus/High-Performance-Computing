#include "MNIST-Loader.h"
#include <iostream>
#include <unistd.h> // for getcwd

#define MNIST_DATA_DIR "datasets/MNIST/"
#define TRAIN_IMAGES MNIST_DATA_DIR "train-images-idx3-ubyte"
#define TRAIN_LABELS MNIST_DATA_DIR "train-labels-idx1-ubyte"
#define TEST_IMAGES MNIST_DATA_DIR "t10k-images-idx3-ubyte"
#define TEST_LABELS MNIST_DATA_DIR "t10k-labels-idx1-ubyte"

int main() {
    MNISTLoader loader(TRAIN_IMAGES, TRAIN_LABELS);

    int num = loader.getNumImages();
    int size = loader.getImageSize();
    const std::vector<float> &images = loader.getImages();
    const std::vector<unsigned char> &labels = loader.getLabels();

    std::cout << "Loaded " << num << " images of size " << size << " pixels each.\n";

    // Example: Copy first image to device, etc...
}
