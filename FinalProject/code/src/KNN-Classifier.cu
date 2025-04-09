#include "KNN-Classifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#define BATCH_SIZE 1000   // Adjust based on GPU memory
#define USE_SHARED_MEMORY true
#define USE_BATCHING true

// Helper function for CUDA error checking
inline void checkCudaError(cudaError_t status, const char* errorMsg) {
	if (status != cudaSuccess) {
		std::cerr << errorMsg << ": " << cudaGetErrorString(status) << std::endl;
		exit(1);
	}
}

// CUDA kernel for computing Euclidean distances
__global__ void computeDistancesKernel(
    float* trainImages, float* testImage, float* distances, int numTrainImages, int imageSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numTrainImages) {
		float sum = 0.0f;

		// Process in chunks to improve memory access patterns
		for (int i = 0; i < imageSize; i++) {
			float diff = trainImages[idx * imageSize + i] - testImage[i];
			sum += diff * diff;
		}

		distances[idx] = sqrt(sum);
	}
}

// CUDA kernel for computing Euclidean distances with shared memory optimization
__global__ void computeDistancesSharedKernel(
    float* trainImages, float* testImage, float* distances, int numTrainImages, int imageSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Use shared memory to cache the test image for faster access
	extern __shared__ float sharedTestImage[];

	// Cooperatively load test image into shared memory
	for (int i = threadIdx.x; i < imageSize; i += blockDim.x) {
		if (i < imageSize) {
			sharedTestImage[i] = testImage[i];
		}
	}
	__syncthreads();

	if (idx < numTrainImages) {
		float sum = 0.0f;

		// Process in chunks to improve memory access patterns
		for (int i = 0; i < imageSize; i++) {
			float diff = trainImages[idx * imageSize + i] - sharedTestImage[i];
			sum += diff * diff;
		}

		distances[idx] = sqrt(sum);
	}
}

// CUDA kernel for finding the majority label among k nearest neighbors
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

// CUDA kernel for find the majority label among k nearest neighbors using shared memory optimization
__global__ void findMajorityLabelSharedKernel(unsigned char* trainLabels,
                                              int*           indices,
                                              unsigned char* predictedLabel,
                                              int            k) {
	// Using shared memory for label counts - much faster for larger k values
	__shared__ int labelCounts[10];   // Assuming max 10 classes for MNIST/CIFAR

	// Initialize shared memory
	if (threadIdx.x < 10) {
		labelCounts[threadIdx.x] = 0;
	}
	__syncthreads();

	// Parallel counting of labels
	for (int i = threadIdx.x; i < k; i += blockDim.x) {
		if (i < k) {
			int           idx   = indices[i];
			unsigned char label = trainLabels[idx];
			atomicAdd(&labelCounts[label], 1);
		}
	}
	__syncthreads();

	// Find label with highest count (only thread 0 does this)
	if (threadIdx.x == 0) {
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
    gpuExecutionTime(0.0),
    gpuMemoryUsage(0.0f) {
	// Set CUDA device
	cudaError_t cudaStatus = cudaSetDevice(deviceId);
	checkCudaError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
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
	cudaMalloc(&d_trainImages, numTrainImages * imageSize * sizeof(float));
	cudaMalloc(&d_trainLabels, numTrainImages * sizeof(unsigned char));
	cudaMalloc(&d_testImage, imageSize * sizeof(float));
	cudaMalloc(&d_distances, numTrainImages * sizeof(float));
	cudaMalloc(&d_indices, numTrainImages * sizeof(int));
	cudaMalloc(&d_predictedLabel, sizeof(unsigned char));

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

void KNNClassifier::train(const std::vector<float>&         trainImages,
                          const std::vector<unsigned char>& trainLabels,
                          const std::string&                datasetName) {
	int numImages = trainLabels.size();
	int imgSize   = trainImages.size() / numImages;

	allocateDeviceMemory(numImages, imgSize);

	cudaMemcpy(d_trainImages,
	           trainImages.data(),
	           numTrainImages * imageSize * sizeof(float),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(d_trainLabels,
	           trainLabels.data(),
	           numTrainImages * sizeof(unsigned char),
	           cudaMemcpyHostToDevice);

	std::cout << "KNN: Loaded " << datasetName << " training data with " << numTrainImages
	          << " images of size " << imageSize << std::endl;
}

void KNNClassifier::computeDistances() {
	// Calculate grid and block dimensions
	int blockSize = 256;
	int gridSize  = (numTrainImages + blockSize - 1) / blockSize;

	if (USE_SHARED_MEMORY) {
		// Calculate shared memory size for the test image
		size_t sharedMemSize = imageSize * sizeof(float);

		// Launch kernel with shared memory
		computeDistancesSharedKernel<<<gridSize, blockSize, sharedMemSize>>>(
			d_trainImages, d_testImage, d_distances, numTrainImages, imageSize);
	}
	else {
		// Launch kernel without shared memory
		computeDistancesKernel<<<gridSize, blockSize>>>(
			d_trainImages, d_testImage, d_distances, numTrainImages, imageSize);
	}

	// Check for kernel launch errors
	checkCudaError(cudaGetLastError(), "computeDistancesKernel launch failed");
}

void KNNClassifier::sortDistancesAndFindMajority() {
	// Initialize indices array (0, 1, 2, ..., numTrainImages-1)
	thrust::device_vector<int> d_idx(numTrainImages);
	thrust::sequence(d_idx.begin(), d_idx.end());

	// Get raw pointers for thrust sorting
	float* thrust_distances = d_distances;
	int*   thrust_indices   = thrust::raw_pointer_cast(d_idx.data());

	// Sort indices by distances on GPU
	thrust::sort_by_key(
	    thrust::device, thrust_distances, thrust_distances + numTrainImages, d_idx.begin());

	// Copy the first k sorted indices back to our device array
	cudaMemcpy(d_indices, thrust_indices, k * sizeof(int), cudaMemcpyDeviceToDevice);

	if (USE_SHARED_MEMORY) {
		// Launch kernel with shared memory
		findMajorityLabelSharedKernel<<<1, 32>>>(d_trainLabels, d_indices, d_predictedLabel, k);
	}
	else {
		// Launch kernel without shared memory
		findMajorityLabelKernel<<<1, 32>>>(d_trainLabels, d_indices, d_predictedLabel, k);
	}

	// Check for kernel launch errors
	checkCudaError(cudaGetLastError(), "findMajorityLabelKernel launch failed");
}

unsigned char KNNClassifier::predict(const std::vector<float>& image, int imageIndex) {
	// Copy test image directly to device
	cudaMemcpy(d_testImage,
	           &image[imageIndex * imageSize],
	           imageSize * sizeof(float),
	           cudaMemcpyHostToDevice);

	// Compute distances between test image and all training images
	computeDistances();

	// Sort distances and find majority label among k nearest neighbors
	sortDistancesAndFindMajority();

	// Copy result back to host
	unsigned char predictedLabel;
	cudaMemcpy(&predictedLabel, d_predictedLabel, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return predictedLabel;
}

void KNNClassifier::predictBatch(const std::vector<float>&   images,
                                 int                         startIndex,
                                 int                         batchSize,
                                 std::vector<unsigned char>& predictions) {
	// Process multiple images in a batch to reduce kernel launch overhead
	for (int i = 0; i < batchSize; i++) {
		int imageIndex = startIndex + i;
		if (imageIndex >= images.size() / imageSize) {
			break;   // Avoid out of bounds access
		}

		// Copy test image to device
		cudaMemcpy(d_testImage,
		           &images[imageIndex * imageSize],
		           imageSize * sizeof(float),
		           cudaMemcpyHostToDevice);

		// Compute distances between test image and all training images
		computeDistances();

		// Sort distances and find majority label among k nearest neighbors
		sortDistancesAndFindMajority();

		// Copy result back to host
		unsigned char predictedLabel;
		cudaMemcpy(
		    &predictedLabel, d_predictedLabel, sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Store the prediction
		predictions[i] = predictedLabel;
	}
}

float KNNClassifier::evaluateDataset(const std::vector<float>&         testImages,
                                     const std::vector<unsigned char>& testLabels) {
	int numTestImages = testLabels.size();
	int correct       = 0;

	// Start Timer
	auto start = std::chrono::high_resolution_clock::now();

    if (USE_BATCHING) {
		// Process images in batches for better performance
		std::vector<unsigned char> batchPredictions(BATCH_SIZE);

		for (int batchStart = 0; batchStart < numTestImages; batchStart += BATCH_SIZE) {
			int currentBatchSize = std::min(BATCH_SIZE, numTestImages - batchStart);
			predictBatch(testImages, batchStart, currentBatchSize, batchPredictions);


			// Count correct predictions
			for (int i = 0; i < currentBatchSize; i++) {
				if (batchPredictions[i] == testLabels[batchStart + i]) {
					correct++;
				}
			}

			// Progress update
			int processed = batchStart + currentBatchSize;
			if (processed % 1000 < BATCH_SIZE || processed == numTestImages) {
				std::cout << "KNN: Processed " << processed << "/" << numTestImages
						<< " test images. Current accuracy: " << (100.0f * correct / processed) << "%"
						<< std::endl;
			}
		}
	}
	else {
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
	}

	// End Timer
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	gpuExecutionTime = elapsed.count();   // in seconds

	float accuracy = 100.0f * correct / numTestImages;
	std::cout << "KNN: Final accuracy: " << accuracy << "%" << std::endl;
	std::cout << "KNN: GPU execution time: " << gpuExecutionTime << " seconds" << std::endl;
	std::cout << "KNN: GPU memory usage: " << gpuMemoryUsage << " MB" << std::endl;

	return accuracy;
}

double KNNClassifier::getGpuExecutionTime() const {
	return gpuExecutionTime;
}

float KNNClassifier::getGpuMemoryUsage() const {
	return gpuMemoryUsage;
}

std::string KNNClassifier::getGpuType() const {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	return std::string(deviceProp.name);
}

int KNNClassifier::getGpuCount() const {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
}