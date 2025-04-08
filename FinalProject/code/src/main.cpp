#include "CIFAR-Loader.h"
#include "KNN-Classifier.h"
#include "MNIST-Loader.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define K_MNIST 1
#define K_CIFAR 1

void saveMetrics(const std::string &dataset,
                 int                k,
                 const std::string &gpuType,
                 int                gpuCount,
                 double             totalExecutionTime,
                 double             gpuExecutionTime,
                 float              memoryUsage,
                 float              accuracy) {
	std::ofstream outFile("output/knn_metrics.csv", std::ios::app);

	// Write header if file is empty
	if (outFile.tellp() == 0) {
		outFile
		    << "Dataset,K,GPUType,GPUCount,TotalExecutionTime(s),GPUExecutionTime(s),MemoryUsage(MB),Accuracy(%)"
		    << std::endl;
	}

	outFile << dataset << "," << k << "," << gpuType << "," << gpuCount << "," << std::fixed
	        << std::setprecision(4) << totalExecutionTime << "," << std::fixed
	        << std::setprecision(4) << gpuExecutionTime << "," << std::fixed << std::setprecision(4)
	        << memoryUsage << "," << std::fixed << std::setprecision(4) << accuracy << std::endl;

	outFile.close();
}

int main() {
	// Test with MNIST
	std::cout << std::endl;
	std::cout << "=========================================================" << std::endl;
	std::cout << " Testing KNN on MNIST dataset with K=" << K_CIFAR << " nearest neighbors"
	          << std::endl;
	std::cout << "=========================================================" << std::endl;

	// Start timing the entire MNIST process
	auto mnist_start_time = std::chrono::high_resolution_clock::now();

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
	float mnist_accuracy = mnist_knn.evaluateDatasetBatched(mnist_test_images, mnist_test_labels);

	// End timing and calculate total time
	auto   mnist_end_time = std::chrono::high_resolution_clock::now();
	double mnist_total_time =
	    std::chrono::duration<double>(mnist_end_time - mnist_start_time).count();
	std::cout << "MNIST: Total dataset processing time: " << mnist_total_time << " seconds"
	          << std::endl;
	std::cout << "MNIST: GPU-only execution time: " << mnist_knn.getGpuExecutionTime() << " seconds"
	          << std::endl;
	std::cout << "MNIST: Non-GPU overhead time: "
	          << (mnist_total_time - mnist_knn.getGpuExecutionTime()) << " seconds" << std::endl;

	// Save metrics
	saveMetrics("MNIST",
	            K_MNIST,
	            mnist_knn.getGpuType(),
	            mnist_knn.getGpuCount(),
	            mnist_total_time,                  // Total time including data loading
	            mnist_knn.getGpuExecutionTime(),   // GPU-specific execution time
	            mnist_knn.getGpuMemoryUsage(),
	            mnist_accuracy);

	// ------------------------------------------------------------------------------------------

	// Test with CIFAR
	std::cout << std::endl;
	std::cout << "=========================================================" << std::endl;
	std::cout << " Testing KNN on CIFAR dataset with K=" << K_CIFAR << " nearest neighbors"
	          << std::endl;
	std::cout << "=========================================================" << std::endl;

	// Start timing the entire CIFAR process
	auto cifar_start_time = std::chrono::high_resolution_clock::now();

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
	float cifar_accuracy = cifar_knn.evaluateDatasetBatched(cifar_test_images, cifar_test_labels);

	// End timing and calculate total time
	auto   cifar_end_time = std::chrono::high_resolution_clock::now();
	double cifar_total_time =
	    std::chrono::duration<double>(cifar_end_time - cifar_start_time).count();
	std::cout << "CIFAR: Total dataset processing time: " << cifar_total_time << " seconds"
	          << std::endl;
	std::cout << "CIFAR: GPU-only execution time: " << cifar_knn.getGpuExecutionTime() << " seconds"
	          << std::endl;
	std::cout << "CIFAR: Non-GPU overhead time: "
	          << (cifar_total_time - cifar_knn.getGpuExecutionTime()) << " seconds" << std::endl;

	/* ------------------------------------------------------------------------------------------ */

	saveMetrics("CIFAR",
	            K_CIFAR,
	            cifar_knn.getGpuType(),
	            cifar_knn.getGpuCount(),
	            cifar_total_time,                  // Total time including data loading
	            cifar_knn.getGpuExecutionTime(),   // GPU-specific execution time
	            cifar_knn.getGpuMemoryUsage(),
	            cifar_accuracy);

	return 0;
}