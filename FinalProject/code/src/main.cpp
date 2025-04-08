#include "CIFAR-Loader.h"
#include "KNN-Classifier.h"
#include "MNIST-Loader.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define K_MNIST 1
#define K_CIFAR 1

void saveMetrics(
    const std::string &dataset, int k, double executionTime, float accuracy, float memoryUsage) {
	std::ofstream outFile("output/knn_metrics.csv", std::ios::app);

	// Write header if file is empty
	if (outFile.tellp() == 0) {
		outFile << "Dataset,K,ExecutionTime(s),Accuracy(%),MemoryUsage(MB)" << std::endl;
	}

	outFile << dataset << "," << k << "," << std::fixed << std::setprecision(4) << executionTime
	        << "," << accuracy << "," << memoryUsage << std::endl;

	outFile.close();
}

int main() {
	// Test with MNIST
	std::cout << std::endl;
	std::cout << "==============================" << std::endl;
	std::cout << " Testing KNN on MNIST dataset" << std::endl;
	std::cout << "==============================" << std::endl;

	MNISTLoader mnist;
	const auto &mnist_train_images = mnist.getTrainImages();
	const auto &mnist_train_labels = mnist.getTrainLabels();
	const auto &mnist_test_images  = mnist.getTestImages();
	const auto &mnist_test_labels  = mnist.getTestLabels();

	// Create and train KNN classifier for MNIST
	KNNClassifier mnist_knn(K_MNIST);
	mnist_knn.train(mnist_train_images, mnist_train_labels, "MNIST");

	// Test on a single image first
	int           mnist_idx       = 0;
	unsigned char predicted_label = mnist_knn.predict(mnist_test_images, mnist_idx);
	std::cout << "MNIST Test Image Actual Label: " << (int) mnist_test_labels[mnist_idx] << "\n";
	std::cout << "MNIST Test Image Predicted Label: " << (int) predicted_label << "\n";
	mnist.writeImageToPPM(mnist_test_images, mnist_idx, "output/mnist_image.ppm");

	// Evaluate on full test set
	float mnist_accuracy = mnist_knn.evaluateAccuracy(mnist_test_images, mnist_test_labels);

	// Save metrics
	saveMetrics("MNIST",
	            K_MNIST,
	            mnist_knn.getLastExecutionTime(),
	            mnist_accuracy,
	            mnist_knn.getGpuMemoryUsage());

	// ------------------------------------------------------------------------------------------

	// Test with CIFAR
	std::cout << std::endl;
	std::cout << "==============================" << std::endl;
	std::cout << " Testing KNN on CIFAR dataset" << std::endl;
	std::cout << "==============================" << std::endl;

	CIFARLoader cifar;
	const auto &cifar_train_images = cifar.getTrainImages();
	const auto &cifar_train_labels = cifar.getTrainLabels();
	const auto &cifar_test_images  = cifar.getTestImages();
	const auto &cifar_test_labels  = cifar.getTestLabels();

	// Create and train KNN classifier for CIFAR
	KNNClassifier cifar_knn(K_CIFAR);
	cifar_knn.train(cifar_train_images, cifar_train_labels, "CIFAR");

	// Test on a single image first
	int cifar_idx   = 0;
	predicted_label = cifar_knn.predict(cifar_test_images, cifar_idx);
	std::cout << "CIFAR Test Image Actual Label: " << (int) cifar_test_labels[cifar_idx] << " ("
	          << cifar.getLabelName(cifar_test_labels[cifar_idx]) << ")\n";
	std::cout << "CIFAR Test Image Predicted Label: " << (int) predicted_label << " ("
	          << cifar.getLabelName(predicted_label) << ")\n";
	cifar.writeImageToPPM(cifar_test_images, cifar_idx, "output/cifar_image.ppm");

	// Evaluate on full test set
	float cifar_accuracy = cifar_knn.evaluateAccuracy(cifar_test_images, cifar_test_labels);

	/* ------------------------------------------------------------------------------------------ */

	saveMetrics("CIFAR",
	            K_CIFAR,
	            cifar_knn.getLastExecutionTime(),
	            cifar_accuracy,
	            cifar_knn.getGpuMemoryUsage());

	return 0;
}