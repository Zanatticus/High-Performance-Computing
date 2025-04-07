#include "KNN-Classifier.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// CUDA kernel for computing Euclidean distances
__global__ void computeDistancesKernel(
    float* trainImages, float* testImage, float* distances, int numTrainImages, int imageSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numTrainImages) {
		float sum = 0.0f;
		for (int i = 0; i < imageSize; i++) {
			float diff = trainImages[idx * imageSize + i] - testImage[i];
			sum += diff * diff;
		}
		distances[idx] = sqrt(sum);
	}
}

// CUDA kernel for counting occurrences of each label within k neighbors
__global__ void findMajorityLabelKernel(unsigned char* trainLabels,
                                        int*           indices,
                                        unsigned char* predictedLabel,
                                        int            k) {
	// This is a simple kernel that runs on a single thread
	// Could be optimized with shared memory for larger k values
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		int labelCounts[10] = {0};   // Assuming max 10 classes for MNIST/CIFAR

		// Count occurrences of each label within k neighbors
		for (int i = 0; i < k; i++) {
			int           idx   = indices[i];
			unsigned char label = trainLabels[idx];
			labelCounts[label]++;
		}

		// Find label with highest count
		int           maxCount      = -1;
		unsigned char majorityLabel = 0;

		for (unsigned char label = 0; label < 10; label++) {
			if (labelCounts[label] > maxCount) {
				maxCount      = labelCounts[label];
				majorityLabel = label;
			}
		}

		*predictedLabel = majorityLabel;
	}
}

// Constructor
KNNClassifier::KNNClassifier(int k, int deviceId) :
    k(k),
    deviceId(deviceId),
    d_trainImages(nullptr),
    d_trainLabels(nullptr),
    d_testImage(nullptr),
    d_distances(nullptr),
    d_indices(nullptr),
    d_predictedLabel(nullptr),
    numTrainImages(0),
    imageSize(0),
    executionTime(0.0),
    gpuMemoryUsage(0.0f) {
	// Set CUDA device
	cudaError_t cudaStatus = cudaSetDevice(deviceId);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
		exit(1);
	}
}

KNNClassifier::~KNNClassifier() {
	freeDeviceMemory();
}

void KNNClassifier::allocateDeviceMemory(int numImages, int imgSize) {
	// Free previous memory if any
	freeDeviceMemory();

	numTrainImages = numImages;
	imageSize      = imgSize;

	// Record memory usage before allocation
	size_t free_before, total;
	cudaMemGetInfo(&free_before, &total);

	// Allocate device memory
	cudaMalloc((void**) &d_trainImages, numTrainImages * imageSize * sizeof(float));
	cudaMalloc((void**) &d_trainLabels, numTrainImages * sizeof(unsigned char));
	cudaMalloc((void**) &d_testImage, imageSize * sizeof(float));
	cudaMalloc((void**) &d_distances, numTrainImages * sizeof(float));
	cudaMalloc((void**) &d_indices, numTrainImages * sizeof(int));
	cudaMalloc((void**) &d_predictedLabel, sizeof(unsigned char));

	// Record memory usage after allocation
	size_t free_after;
	cudaMemGetInfo(&free_after, &total);
	gpuMemoryUsage = (free_before - free_after) / (1024.0f * 1024.0f);   // in MB
}

void KNNClassifier::freeDeviceMemory() {
	if (d_trainImages)
		cudaFree(d_trainImages);
	if (d_trainLabels)
		cudaFree(d_trainLabels);
	if (d_testImage)
		cudaFree(d_testImage);
	if (d_distances)
		cudaFree(d_distances);
	if (d_indices)
		cudaFree(d_indices);
	if (d_predictedLabel)
		cudaFree(d_predictedLabel);

	d_trainImages    = nullptr;
	d_trainLabels    = nullptr;
	d_testImage      = nullptr;
	d_distances      = nullptr;
	d_indices        = nullptr;
	d_predictedLabel = nullptr;
}

void KNNClassifier::trainMNIST(const std::vector<float>&         trainImages,
                               const std::vector<unsigned char>& trainLabels) {
	int numImages = trainLabels.size();
	int imgSize   = trainImages.size() / numImages;

	allocateDeviceMemory(numImages, imgSize);

	// Copy training data to device
	cudaMemcpy(d_trainImages,
	           trainImages.data(),
	           numTrainImages * imageSize * sizeof(float),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_trainLabels,
	           trainLabels.data(),
	           numTrainImages * sizeof(unsigned char),
	           cudaMemcpyHostToDevice);

	std::cout << "KNN: Loaded MNIST training data with " << numTrainImages << " images of size "
	          << imageSize << std::endl;
}

void KNNClassifier::trainCIFAR(const std::vector<float>&         trainImages,
                               const std::vector<unsigned char>& trainLabels) {
	int numImages = trainLabels.size();
	int imgSize   = trainImages.size() / numImages;

	allocateDeviceMemory(numImages, imgSize);

	// Copy training data to device
	cudaMemcpy(d_trainImages,
	           trainImages.data(),
	           numTrainImages * imageSize * sizeof(float),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_trainLabels,
	           trainLabels.data(),
	           numTrainImages * sizeof(unsigned char),
	           cudaMemcpyHostToDevice);

	std::cout << "KNN: Loaded CIFAR training data with " << numTrainImages << " images of size "
	          << imageSize << std::endl;
}

void KNNClassifier::computeDistances(int numTestImages) {
	// Configure grid and block dimensions
	int threadsPerBlock = 256;
	int blocksPerGrid   = (numTrainImages + threadsPerBlock - 1) / threadsPerBlock;

	// Launch kernel
	computeDistancesKernel<<<blocksPerGrid, threadsPerBlock>>>(
	    d_trainImages, d_testImage, d_distances, numTrainImages, imageSize);

	// Check for kernel launch errors
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "computeDistancesKernel launch failed: " << cudaGetErrorString(cudaStatus)
		          << std::endl;
		exit(1);
	}
}

void KNNClassifier::sortDistancesAndFindMajority() {
	// Initialize indices array (0, 1, 2, ..., numTrainImages-1)
	thrust::device_vector<int> d_idx(numTrainImages);
	thrust::sequence(d_idx.begin(), d_idx.end());

	// Get raw pointers for thrust sorting
	float* thrust_distances = d_distances;
	int*   thrust_indices   = thrust::raw_pointer_cast(d_idx.data());

	// Sort indices by distances
	thrust::sort_by_key(
	    thrust::device, thrust_distances, thrust_distances + numTrainImages, d_idx.begin());

	// Copy the first k sorted indices back to our device array
	cudaMemcpy(d_indices, thrust_indices, k * sizeof(int), cudaMemcpyDeviceToDevice);

	// Find majority label
	findMajorityLabelKernel<<<1, 1>>>(d_trainLabels, d_indices, d_predictedLabel, k);

	// Check for kernel launch errors
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "findMajorityLabelKernel launch failed: " << cudaGetErrorString(cudaStatus)
		          << std::endl;
		exit(1);
	}
}

unsigned char KNNClassifier::predict(const std::vector<float>& image, int imageIndex) {
	// Time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Extract single image and copy to device
	std::vector<float> singleImage(imageSize);
	for (int i = 0; i < imageSize; i++) {
		singleImage[i] = image[imageIndex * imageSize + i];
	}

	cudaMemcpy(d_testImage, singleImage.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);

	// Compute distances between test image and all training images
	computeDistances(1);

	// Sort distances and find majority label among k nearest neighbors
	sortDistancesAndFindMajority();

	// Copy result back to host
	unsigned char predictedLabel;
	cudaMemcpy(&predictedLabel, d_predictedLabel, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Time measurement end
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	executionTime = milliseconds / 1000.0;   // Convert from ms to seconds

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return predictedLabel;
}

float KNNClassifier::evaluateAccuracy(const std::vector<float>&         testImages,
                                      const std::vector<unsigned char>& testLabels) {
	int numTestImages = testLabels.size();
	int correct       = 0;

	// Time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// For each test image
	for (int i = 0; i < numTestImages; i++) {
		unsigned char predictedLabel = predict(testImages, i);
		if (predictedLabel == testLabels[i]) {
			correct++;
		}

		// Progress update every 1000 images
		if ((i + 1) % 1000 == 0 || i == numTestImages - 1) {
			std::cout << "KNN: Processed " << (i + 1) << "/" << numTestImages
			          << " test images. Current accuracy: " << (100.0f * correct / (i + 1)) << "%"
			          << std::endl;
		}
	}

	// Time measurement end
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	executionTime = milliseconds / 1000.0;   // Convert from ms to seconds

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	float accuracy = 100.0f * correct / numTestImages;
	std::cout << "KNN: Final accuracy: " << accuracy << "%" << std::endl;
	std::cout << "KNN: Total evaluation time: " << executionTime << " seconds" << std::endl;
	std::cout << "KNN: GPU memory usage: " << gpuMemoryUsage << " MB" << std::endl;

	return accuracy;
}

void KNNClassifier::setK(int newK) {
	k = newK;
}

int KNNClassifier::getK() const {
	return k;
}

double KNNClassifier::getLastExecutionTime() const {
	return executionTime;
}

float KNNClassifier::getGpuMemoryUsage() const {
	return gpuMemoryUsage;
}