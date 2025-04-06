#include "MNIST-Loader.h"
#include "CIFAR-Loader.h"
#include <iostream>
#include <fstream>
#include <vector>

void write_mnist_ppm(const float* image, int width, int height, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    out << "P6\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; ++i) {
        unsigned char gray = static_cast<unsigned char>(image[i] * 255);
        out.put(gray); // R
        out.put(gray); // G
        out.put(gray); // B
    }

    out.close();
    std::cout << "Saved MNIST image to " << filename << std::endl << std::endl;
}


void write_cifar_ppm(const float *image, int width, int height, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    out << "P6\n" << width << " " << height << "\n255\n";

    // CIFAR stores channels as [R(1024), G(1024), B(1024)]
    int channelSize = width * height;
    const float *red = image;
    const float *green = image + channelSize;
    const float *blue = image + 2 * channelSize;

    for (int i = 0; i < channelSize; ++i) {
        out.put(static_cast<unsigned char>(red[i] * 255));
        out.put(static_cast<unsigned char>(green[i] * 255));
        out.put(static_cast<unsigned char>(blue[i] * 255));
    }

    out.close();
    std::cout << "Saved CIFAR image to " << filename << std::endl << std::endl;
}

int main() {
    MNISTLoader mnist;
    const auto &mnist_test_images = mnist.getTestImages();
    const auto &mnist_test_labels = mnist.getTestLabels();
    const auto &mnist_train_images = mnist.getTrainImages();
    const auto &mnist_train_labels = mnist.getTrainLabels();

    std::cout << "MNIST Train Size: " << mnist.getNumTrainImages() << "\n";
    std::cout << "MNIST Test Size: " << mnist.getNumTestImages() << "\n";

    int mnist_idx = 0;
    std::cout << "MNIST Test Image Label: " << (int)mnist_test_labels[mnist_idx] << "\n";
    write_mnist_ppm(&mnist_test_images[mnist_idx * mnist.getImageSize()], 28, 28, "output/mnist_image.ppm");



    CIFARLoader cifar;
    const auto &cifar_test_images = cifar.getTestImages();
    const auto &cifar_test_labels = cifar.getTestLabels();
    const auto &cifar_train_images = cifar.getTrainImages();
    const auto &cifar_train_labels = cifar.getTrainLabels();

    std::cout << "CIFAR Train Size: " << cifar.getNumTrainImages() << "\n";
    std::cout << "CIFAR Test Size: " << cifar.getNumTestImages() << "\n";

    int cifar_idx = 0;   
    std::cout << "Label: " << (int)cifar_test_labels[cifar_idx] << " (" << cifar.getLabelName(cifar_test_labels[cifar_idx]) << ")\n";
    write_cifar_ppm(&cifar_test_images[cifar_idx * cifar.getImageSize()], cifar.getWidth(), cifar.getHeight(), "output/cifar_img.ppm");
    

    // Example: Copy first image to device, etc...
}
