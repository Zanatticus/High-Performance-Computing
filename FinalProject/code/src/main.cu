#include "MNIST-Loader.h"
#include <iostream>

#define MNIST_DATA_DIR "datasets/MNIST/"
#define TRAIN_IMAGES MNIST_DATA_DIR "train-images.idx3-ubyte"
#define TRAIN_LABELS MNIST_DATA_DIR "train-labels.idx1-ubyte"
#define TEST_IMAGES MNIST_DATA_DIR "t10k-images.idx3-ubyte"
#define TEST_LABELS MNIST_DATA_DIR "t10k-labels.idx1-ubyte"


// Displays the MNIST images in a simple ASCII format
void printImage(const float *image, int width, int height) {
    const char shades[] = " .:-=+*#%@";  // 10 levels of intensity
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float pixel = image[row * width + col];
            int shadeIndex = static_cast<int>(pixel * 9);  // map 0.0–1.0 to 0–9
            std::cout << shades[shadeIndex];
        }
        std::cout << "\n";
    }
}

int main() {
    MNISTLoader loader(TRAIN_IMAGES, TRAIN_LABELS);

    int num = loader.getNumImages();
    int size = loader.getImageSize();
    const std::vector<float> &images = loader.getImages();
    const std::vector<unsigned char> &labels = loader.getLabels();

    std::cout << "Loaded " << num << " images of size " << size << " pixels each.\n";

    for (int i = 0; i < 1; ++i) {
        std::cout << "\n=== Image #" << i << " | Label: " << static_cast<int>(labels[i]) << " ===\n";
        printImage(&images[i * size], 28, 28);
    }


    // Example: Copy first image to device, etc...
}
