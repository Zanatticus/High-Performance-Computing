#include "CIFAR-Loader.h"
#include "KNN-Classifier.h"
#include "MNIST-Loader.h"
#include "STL-Loader.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define K_MNIST 5
#define K_CIFAR 5
#define K_STL   5

#define TEST_MNIST true
#define TEST_CIFAR true
#define TEST_STL   true

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
		outFile << "Dataset,K,GPUType,GPUCount,TotalExecutionTime(s),GPUExecutionTime(s),MemoryUsage(MB),Accuracy(%)"
		        << std::endl;
	}

	outFile << dataset << "," << k << "," << gpuType << "," << gpuCount << "," << std::fixed << std::setprecision(4)
	        << totalExecutionTime << "," << std::fixed << std::setprecision(4) << gpuExecutionTime << "," << std::fixed
	        << std::setprecision(4) << memoryUsage << "," << std::fixed << std::setprecision(4) << accuracy
	        << std::endl;

	outFile.close();
}

int main() {
	if (TEST_MNIST) {
		try {
			// Test with MNIST
			std::cout << std::endl;
			std::cout << "=========================================================" << std::endl;
			std::cout << " Testing KNN on MNIST dataset with K=" << K_MNIST << " nearest neighbors" << std::endl;
			std::cout << "=========================================================" << std::endl;

			// Start timing the entire MNIST process
			auto mnist_start_time = std::chrono::high_resolution_clock::now();

			MNISTLoader mnist;
			const auto &mnist_train_images = mnist.getTrainImages();
			const auto &mnist_train_labels = mnist.getTrainLabels();
			const auto &mnist_test_images  = mnist.getTestImages();
			const auto &mnist_test_labels  = mnist.getTestLabels();

			// Create and train KNN classifier for MNIST
			KNNClassifier mnist_knn(
			    mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels, "MNIST", K_MNIST);
			mnist_knn.train();

			// Test on a single image first
			int           mnist_idx       = 0;
			unsigned char predicted_label = mnist_knn.predict(mnist_idx);
			std::cout << "MNIST Test Image Actual Label: " << (int) mnist_test_labels[mnist_idx] << "\n";
			std::cout << "MNIST Test Image Predicted Label: " << (int) predicted_label << "\n";
			mnist.writeImageToPPM(mnist_test_images, mnist_idx, "output/mnist_image.ppm");

			// Evaluate on full test set
			float mnist_accuracy = mnist_knn.evaluateDataset();

			// End timing and calculate total time
			auto   mnist_end_time   = std::chrono::high_resolution_clock::now();
			double mnist_total_time = std::chrono::duration<double>(mnist_end_time - mnist_start_time).count();
			std::cout << "MNIST: Total dataset processing time: " << mnist_total_time << " seconds" << std::endl;
			std::cout << "MNIST: GPU-only execution time: " << mnist_knn.getGpuExecutionTime() << " seconds"
			          << std::endl;
			std::cout << "MNIST: Non-GPU overhead time: " << (mnist_total_time - mnist_knn.getGpuExecutionTime())
			          << " seconds" << std::endl;

			// Save metrics
			saveMetrics("MNIST",
			            K_MNIST,
			            mnist_knn.getGpuType(),
			            mnist_knn.getGpuCount(),
			            mnist_total_time,                  // Total time including data loading
			            mnist_knn.getGpuExecutionTime(),   // GPU-specific execution time
			            mnist_knn.getGpuMemoryUsage(),
			            mnist_accuracy);
		} catch (const std::exception &e) { std::cerr << "ERROR USING MNIST DATASET: " << e.what() << std::endl; }
	}

	/* ------------------------------------------------------------------------------------------------- */

	if (TEST_CIFAR) {
		try {
			// Test with CIFAR-10
			std::cout << std::endl;
			std::cout << "============================================================" << std::endl;
			std::cout << " Testing KNN on CIFAR-10 dataset with K=" << K_CIFAR << " nearest neighbors" << std::endl;
			std::cout << "============================================================" << std::endl;

			// Start timing the entire CIFAR-10 process
			auto cifar_start_time = std::chrono::high_resolution_clock::now();

			CIFARLoader cifar;
			const auto &cifar_train_images = cifar.getTrainImages();
			const auto &cifar_train_labels = cifar.getTrainLabels();
			const auto &cifar_test_images  = cifar.getTestImages();
			const auto &cifar_test_labels  = cifar.getTestLabels();

			// Create and train KNN classifier for CIFAR-10
			KNNClassifier cifar_knn(
			    cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, "CIFAR-10", K_CIFAR);
			cifar_knn.train();

			// Test on a single image first
			int           cifar_idx       = 0;
			unsigned char predicted_label = cifar_knn.predict(cifar_idx);
			std::cout << "CIFAR-10 Test Image Actual Label: " << (int) cifar_test_labels[cifar_idx] << " ("
			          << cifar.getLabelName(cifar_test_labels[cifar_idx]) << ")\n";
			std::cout << "CIFAR-10 Test Image Predicted Label: " << (int) predicted_label << " ("
			          << cifar.getLabelName(predicted_label) << ")\n";
			cifar.writeImageToPPM(cifar_test_images, cifar_idx, "output/cifar_image.ppm");

			// Evaluate on full test set
			float cifar_accuracy = cifar_knn.evaluateDataset();

			// End timing and calculate total time
			auto   cifar_end_time   = std::chrono::high_resolution_clock::now();
			double cifar_total_time = std::chrono::duration<double>(cifar_end_time - cifar_start_time).count();
			std::cout << "CIFAR-10: Total dataset processing time: " << cifar_total_time << " seconds" << std::endl;
			std::cout << "CIFAR-10: GPU-only execution time: " << cifar_knn.getGpuExecutionTime() << " seconds"
			          << std::endl;
			std::cout << "CIFAR-10: Non-GPU overhead time: " << (cifar_total_time - cifar_knn.getGpuExecutionTime())
			          << " seconds" << std::endl;

			// Save metrics
			saveMetrics("CIFAR-10",
			            K_CIFAR,
			            cifar_knn.getGpuType(),
			            cifar_knn.getGpuCount(),
			            cifar_total_time,                  // Total time including data loading
			            cifar_knn.getGpuExecutionTime(),   // GPU-specific execution time
			            cifar_knn.getGpuMemoryUsage(),
			            cifar_accuracy);
		} catch (const std::exception &e) { std::cerr << "ERROR USING CIFAR-10 DATASET: " << e.what() << std::endl; }
	}

	/* ----------------------------------------------------------------------------------------------- */

	if (TEST_STL) {
		try {
			// Test with STL-10
			std::cout << std::endl;
			std::cout << "==========================================================" << std::endl;
			std::cout << " Testing KNN on STL-10 dataset with K=" << K_STL << " nearest neighbors" << std::endl;
			std::cout << "==========================================================" << std::endl;

			// Start timing the entire STL-10 process
			auto stl_start_time = std::chrono::high_resolution_clock::now();

			STLLoader   stl;
			const auto &stl_train_images = stl.getTrainImages();
			const auto &stl_train_labels = stl.getTrainLabels();
			const auto &stl_test_images  = stl.getTestImages();
			const auto &stl_test_labels  = stl.getTestLabels();

			// Create and train KNN classifier for STL-10
			KNNClassifier stl_knn(
			    stl_train_images, stl_train_labels, stl_test_images, stl_test_labels, "STL-10", K_STL);
			stl_knn.train();

			// Test on a single image first
			int           stl_idx         = 0;
			unsigned char predicted_label = stl_knn.predict(stl_idx);
			std::cout << "STL-10 Test Image Actual Label: " << (int) stl_test_labels[stl_idx] << "\n";
			std::cout << "STL-10 Test Image Predicted Label: " << (int) predicted_label << "\n";
			stl.writeImageToPPM(stl_test_images, stl_idx, "output/stl_image.ppm");

			// Evaluate on full test set
			float stl_accuracy = stl_knn.evaluateDataset();

			// End timing and calculate total time
			auto   stl_end_time   = std::chrono::high_resolution_clock::now();
			double stl_total_time = std::chrono::duration<double>(stl_end_time - stl_start_time).count();

			std::cout << "STL-10: Total dataset processing time: " << stl_total_time << " seconds\n";
			std::cout << "STL-10: GPU-only execution time: " << stl_knn.getGpuExecutionTime() << " seconds\n";
			std::cout << "STL-10: Non-GPU overhead time: " << (stl_total_time - stl_knn.getGpuExecutionTime())
			          << " seconds\n";

			// Save metrics
			saveMetrics("STL-10",
			            K_STL,
			            stl_knn.getGpuType(),
			            stl_knn.getGpuCount(),
			            stl_total_time,
			            stl_knn.getGpuExecutionTime(),
			            stl_knn.getGpuMemoryUsage(),
			            stl_accuracy);
		} catch (const std::exception &e) { std::cerr << "ERROR USING STL-10 DATASET: " << e.what() << std::endl; }
	}

	/* ------------------------------------------------------------------------------------------------- */

	return 0;
}