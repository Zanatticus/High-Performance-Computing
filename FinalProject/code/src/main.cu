#include "MNIST-Loader.h"
#include "CIFAR-Loader.h"
#include <iostream>
#include <fstream>
#include <vector>

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
    mnist.writeImageToPPM(mnist_test_images, mnist_idx, "output/mnist_image.ppm") ;

    CIFARLoader cifar;
    const auto &cifar_test_images = cifar.getTestImages();
    const auto &cifar_test_labels = cifar.getTestLabels();
    const auto &cifar_train_images = cifar.getTrainImages();
    const auto &cifar_train_labels = cifar.getTrainLabels();

    std::cout << "CIFAR Train Size: " << cifar.getNumTrainImages() << "\n";
    std::cout << "CIFAR Test Size: " << cifar.getNumTestImages() << "\n";

    int cifar_idx = 0;   
    std::cout << "CIFAR Test Image Label: " << (int)cifar_test_labels[cifar_idx] << " (" << cifar.getLabelName(cifar_test_labels[cifar_idx]) << ")\n";
    cifar.writeImageToPPM(cifar_test_images, cifar_idx, "output/cifar_image.ppm");
    
    // Example: Copy first image to device, etc...
}
