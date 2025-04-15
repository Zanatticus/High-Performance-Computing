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

// Helper function for CUDA error checking
inline void checkCudaError(cudaError_t status, const char* errorMsg) {
	if (status != cudaSuccess) {
		if (errorMsg != nullptr) {
			std::cerr << errorMsg << ": ";
		}
		std::cerr << cudaGetErrorString(status) << std::endl;
		exit(1);
	}
}

// Original baseline kernel for computing Euclidean distances
__global__ void
computeDistancesKernel(float* trainImages, float* testImage, float* distances, int numTrainImages, int imageSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numTrainImages) {
		float sum = 0.0f;

		for (int i = 0; i < imageSize; i++) {
			float diff = trainImages[idx * imageSize + i] - testImage[i];
			sum += diff * diff;
		}

		distances[idx] = sqrtf(sum);
	}
}

// New batched kernel that processes all test images against training images in shared memory
__global__ void computeDistancesBatchedKernel(
    float* trainImages, float* testImages, float* allDistances,
    int numTrainImages, int numTestImages, int imageSize) {
    
    extern __shared__ float sharedTrain[];
    
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int trainIdx = blockIdx.x;  // Each block handles one training image
    
    // Load one training image into shared memory collaboratively
    for (int i = tid; i < imageSize; i += numThreads) {
        sharedTrain[i] = trainImages[trainIdx * imageSize + i];
    }
    __syncthreads();
    
    // Each thread handles multiple test images
    for (int testIdx = tid; testIdx < numTestImages; testIdx += numThreads) {
        float sum = 0.0f;
        
        // Compute distance between this test image and the training image
        for (int i = 0; i < imageSize; i++) {
            float diff = testImages[testIdx * imageSize + i] - sharedTrain[i];
            sum += diff * diff;
        }
        
        // Store distance in global memory
        // Distance matrix is [testImages][trainImages]
        allDistances[testIdx * numTrainImages + trainIdx] = sqrtf(sum);
    }
}

// CUDA kernel for finding the majority label among k nearest neighbors
__global__ void
findMajorityLabelKernel(unsigned char* trainLabels, int* indices, unsigned char* predictedLabel, int k_neighbors) {
	// Using shared memory for label counts
	__shared__ int labelCounts[10];   // Assuming max 10 classes

	// Initialize shared memory
	if (threadIdx.x < 10) {
		labelCounts[threadIdx.x] = 0;
	}
	__syncthreads();

	// Count labels in parallel
	for (int i = threadIdx.x; i < k_neighbors; i += blockDim.x) {
		int idx = indices[i];
		atomicAdd(&labelCounts[trainLabels[idx]], 1);
	}
	__syncthreads();

	// Find majority label (only thread 0)
	if (threadIdx.x == 0) {
		int maxCount = -1;
		unsigned char majorityLabel = 0;

		for (unsigned char label = 0; label < 10; label++) {
			if (labelCounts[label] > maxCount) {
				maxCount = labelCounts[label];
				majorityLabel = label;
			}
		}

		*predictedLabel = majorityLabel;
	}
}

// Batched majority finding kernel - processes one test image per block
__global__ void findMajorityLabelBatchedKernel(
    unsigned char* trainLabels, int* kIndices, unsigned char* predictions,
    int numTestImages, int k_neighbors) {
    
    int testIdx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (testIdx >= numTestImages) return;
    
    // Using shared memory for label counts
    __shared__ int labelCounts[10];  // Assuming max 10 classes
    
    // Initialize shared memory
    if (tid < 10) {
        labelCounts[tid] = 0;
    }
    __syncthreads();
    
    // Get pointer to the k indices for this test image
    int* indices = kIndices + (testIdx * k_neighbors);
    
    // Count labels in parallel
    for (int i = tid; i < k_neighbors; i += blockDim.x) {
        int idx = indices[i];
        atomicAdd(&labelCounts[trainLabels[idx]], 1);
    }
    __syncthreads();
    
    // Find majority label (only thread 0)
    if (tid == 0) {
        int maxCount = -1;
        unsigned char majorityLabel = 0;
        
        for (unsigned char label = 0; label < 10; label++) {
            if (labelCounts[label] > maxCount) {
                maxCount = labelCounts[label];
                majorityLabel = label;
            }
        }
        
        predictions[testIdx] = majorityLabel;
    }
}

KNNClassifier::KNNClassifier(const std::vector<float>&         trainImages,
                             const std::vector<unsigned char>& trainLabels,
                             const std::vector<float>&         testImages,
                             const std::vector<unsigned char>& testLabels,
                             const std::string&                datasetName,
                             int                               k,
							 bool 							   useBatchMode,
                             int                               deviceId) :
    d_trainImages(nullptr),
    d_trainLabels(nullptr),
    d_testImage(nullptr),
    d_testImages(nullptr),
    d_distances(nullptr),
    d_indices(nullptr),
    d_predictedLabel(nullptr),
    h_trainImages(trainImages),
    h_trainLabels(trainLabels),
    h_testImages(testImages),
    h_testLabels(testLabels),
    datasetName(datasetName),
    k_neighbors(k),
    numTrainImages(trainLabels.size()),
    numTestImages(testLabels.size()),
    imageSize(static_cast<int>(trainImages.size() / trainLabels.size())),
    gpuExecutionTime(0.0),
    gpuMemoryUsage(0.0f),
    useBatchMode(useBatchMode),
    deviceId(deviceId) {
	cudaError_t cudaStatus = cudaSetDevice(deviceId);
	checkCudaError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	if (useBatchMode) {
		allocateDeviceMemoryBatched();
	}
	else {
		allocateDeviceMemory();
	}
}

KNNClassifier::~KNNClassifier() {
	freeDeviceMemory();
}

void KNNClassifier::allocateDeviceMemory() {
	// Free previous memory if any
	freeDeviceMemory();

	// Record memory usage before allocation
	size_t free_before, total;
	cudaMemGetInfo(&free_before, &total);

	// Standard memory allocation for one-test-at-a-time approach
	cudaMalloc(&d_trainImages, numTrainImages * imageSize * sizeof(float));
	cudaMalloc(&d_trainLabels, numTrainImages * sizeof(unsigned char));
	cudaMalloc(&d_testImage, imageSize * sizeof(float));
	cudaMalloc(&d_distances, numTrainImages * sizeof(float));
	cudaMalloc(&d_indices, numTrainImages * sizeof(int));
	cudaMalloc(&d_predictedLabel, sizeof(unsigned char));

	// Record memory usage after allocation
	size_t free_after;
	cudaMemGetInfo(&free_after, &total);
	gpuMemoryUsage = (free_before - free_after) / (1024.0f * 1024.0f);  // in MB
}

void KNNClassifier::allocateDeviceMemoryBatched() {
    // Free previous memory if any
    freeDeviceMemory();
    
    // Record memory usage before allocation
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    
    // Allocate memory for training data
    cudaMalloc(&d_trainImages, numTrainImages * imageSize * sizeof(float));
    cudaMalloc(&d_trainLabels, numTrainImages * sizeof(unsigned char));
    
    // Allocate memory for ALL test images
    cudaMalloc(&d_testImages, numTestImages * imageSize * sizeof(float));
    
    // Allocate memory for the full distance matrix
    cudaMalloc(&d_distances, numTestImages * numTrainImages * sizeof(float));
    
    // Allocate memory for k indices for each test image
    cudaMalloc(&d_indices, numTestImages * k_neighbors * sizeof(int));
    
    // Allocate memory for all predictions
    cudaMalloc(&d_predictedLabel, numTestImages * sizeof(unsigned char));
    
    // Record memory usage after allocation
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);
    gpuMemoryUsage = (free_before - free_after) / (1024.0f * 1024.0f);  // in MB
}

void KNNClassifier::freeDeviceMemory() {
	if (d_trainImages)
		cudaFree(d_trainImages);
	if (d_trainLabels)
		cudaFree(d_trainLabels);
	if (d_testImage)
		cudaFree(d_testImage);
	if (d_testImages)
		cudaFree(d_testImages);
	if (d_distances)
		cudaFree(d_distances);
	if (d_indices)
		cudaFree(d_indices);
	if (d_predictedLabel)
		cudaFree(d_predictedLabel);

	d_trainImages = nullptr;
	d_trainLabels = nullptr;
	d_testImage = nullptr;
	d_testImages = nullptr;
	d_distances = nullptr;
	d_indices = nullptr;
	d_predictedLabel = nullptr;
}

void KNNClassifier::train() {
    if (useBatchMode) {
        trainBatched();
        return;
    }

    // Regular training approach
    allocateDeviceMemory();

    cudaMemcpy(d_trainImages, h_trainImages.data(), sizeof(float) * h_trainImages.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_trainLabels, h_trainLabels.data(), sizeof(unsigned char) * h_trainLabels.size(), cudaMemcpyHostToDevice);

    std::cout << "KNN: Loaded " << datasetName << " training data with " << numTrainImages << " images of size "
              << imageSize << std::endl;
}

void KNNClassifier::trainBatched() {
    useBatchMode = true;
    allocateDeviceMemoryBatched();
    
    // Copy training data to GPU
    cudaMemcpy(d_trainImages, h_trainImages.data(), 
               sizeof(float) * h_trainImages.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainLabels, h_trainLabels.data(), 
               sizeof(unsigned char) * h_trainLabels.size(), cudaMemcpyHostToDevice);
    
    // Copy all test images to GPU at once
    cudaMemcpy(d_testImages, h_testImages.data(), 
               sizeof(float) * h_testImages.size(), cudaMemcpyHostToDevice);
    
    std::cout << "KNN Batched: Loaded " << datasetName 
              << " data with " << numTrainImages << " training images and "
              << numTestImages << " test images of size " << imageSize << std::endl;
}

void KNNClassifier::computeDistances() {
	int gridSize = (numTrainImages + blockSize - 1) / blockSize;
	
	computeDistancesKernel<<<gridSize, blockSize>>>(
	    d_trainImages, d_testImage, d_distances, numTrainImages, imageSize);
}

void KNNClassifier::sortDistancesAndFindMajority() {
	// Initialize indices array (0, 1, 2, ..., numTrainImages-1)
	thrust::device_vector<int> d_idx(numTrainImages);
	thrust::sequence(d_idx.begin(), d_idx.end());

	// Get raw pointers for thrust sorting
	float* thrust_distances = d_distances;
	int* thrust_indices = thrust::raw_pointer_cast(d_idx.data());

	// Sort indices by distances on GPU
	thrust::sort_by_key(thrust::device, thrust_distances, thrust_distances + numTrainImages, d_idx.begin());

	// Copy the first k sorted indices back to our device array
	cudaMemcpy(d_indices, thrust_indices, k_neighbors * sizeof(int), cudaMemcpyDeviceToDevice);

	findMajorityLabelKernel<<<1, blockSize>>>(d_trainLabels, d_indices, d_predictedLabel, k_neighbors);

	// Check for kernel launch errors
	checkCudaError(cudaGetLastError(), "findMajorityLabelKernel launch failed");
}

unsigned char KNNClassifier::predict(int imageIndex) {
	const float* image_ptr = h_testImages.data() + imageIndex * imageSize;

	// Copy test image directly to device
	cudaMemcpy(d_testImage, image_ptr, imageSize * sizeof(float), cudaMemcpyHostToDevice);

	// Compute distances between test image and all training images
	computeDistances();

	// Sort distances and find majority label among k nearest neighbors
	sortDistancesAndFindMajority();

	// Copy result back to host
	unsigned char predictedLabel;
	cudaMemcpy(&predictedLabel, d_predictedLabel, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return predictedLabel;
}

float KNNClassifier::evaluateDataset() {
    if (useBatchMode) {
        return evaluateDatasetBatched();
    }

    // Regular non-batched evaluation
    int correct = 0;

    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();

    // For each test image
    for (int i = 0; i < numTestImages; i++) {
        unsigned char predictedLabel = predict(i);

        if (predictedLabel == h_testLabels[i]) {
            correct++;
        }

        // Progress update every 1000 images
        if ((i + 1) % 1000 == 0 || i == numTestImages - 1) {
            std::cout << "KNN: Processed " << (i + 1) << "/" << numTestImages
                      << " test images. Current accuracy: " << (100.0f * correct / (i + 1)) << "%" << std::endl;
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

float KNNClassifier::evaluateDatasetBatched() {
    // Make sure we're in batch mode
    if (!useBatchMode) {
        std::cerr << "Error: Batch evaluation called but not in batch mode!" << std::endl;
        return 0.0f;
    }
    
    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Compute all distances between all test images and all training images
    int gridSize = numTrainImages; // One block per training image
    
    // Each block loads one training image into shared memory and compares against all test images
    size_t sharedMemSize = imageSize * sizeof(float); // For one training image
	
	// Attempt to dynamically change the maximum shared memory size for the streaming multiprocessor block
	cudaError_t err = cudaFuncSetAttribute(computeDistancesBatchedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

	if (err != cudaSuccess) {
		std::cerr << "Warning: Failed to set shared memory size to " << sharedMemSize 
              << " bytes: " << cudaGetErrorString(err) << std::endl;
		return 0.0f;	
	}

    computeDistancesBatchedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_trainImages, d_testImages, d_distances, 
        numTrainImages, numTestImages, imageSize);
    
    checkCudaError(cudaGetLastError(), "Batched distance kernel launch failed");
    cudaDeviceSynchronize();
    
    // Step 2: Find k nearest neighbors for each test image
    // Allocate host memory for distances (could be large!)
    float* h_allDistances = new float[numTestImages * numTrainImages];
    cudaMemcpy(h_allDistances, d_distances, 
               numTestImages * numTrainImages * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Allocate host memory for k indices for each test image
    int* h_kIndices = new int[numTestImages * k_neighbors];
    
    // For each test image, find k nearest training images
    for (int testIdx = 0; testIdx < numTestImages; testIdx++) {
        float* testDistances = h_allDistances + (testIdx * numTrainImages);
        
        // Create pairs of (distance, index)
        std::vector<std::pair<float, int>> distanceIndices(numTrainImages);
        for (int i = 0; i < numTrainImages; i++) {
            distanceIndices[i] = std::make_pair(testDistances[i], i);
        }
        
        // Sort to find k smallest distances
        std::partial_sort(distanceIndices.begin(), 
                         distanceIndices.begin() + k_neighbors, 
                         distanceIndices.end());
        
        // Store indices of k nearest neighbors
        for (int k = 0; k < k_neighbors; k++) {
            h_kIndices[testIdx * k_neighbors + k] = distanceIndices[k].second;
        }
    }
    
    // Copy k indices to device
    cudaMemcpy(d_indices, h_kIndices, 
               numTestImages * k_neighbors * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    // Step 3: Find majority label for each test image
    findMajorityLabelBatchedKernel<<<numTestImages, blockSize>>>(
        d_trainLabels, d_indices, d_predictedLabel, 
        numTestImages, k_neighbors);
    
    checkCudaError(cudaGetLastError(), "Majority label kernel launch failed");
    cudaDeviceSynchronize();
    
    // Step 4: Copy predictions back to host and calculate accuracy
    unsigned char* h_predictions = new unsigned char[numTestImages];
    cudaMemcpy(h_predictions, d_predictedLabel, 
               numTestImages * sizeof(unsigned char), 
               cudaMemcpyDeviceToHost);
    
    int correct = 0;
    for (int i = 0; i < numTestImages; i++) {
        if (h_predictions[i] == h_testLabels[i]) {
            correct++;
        }
        
        // Progress update every 1000 images
        if ((i + 1) % 1000 == 0 || i == numTestImages - 1) {
            std::cout << "KNN Batched: Processed " << (i + 1) << "/" << numTestImages
                     << " test images. Current accuracy: " 
                     << (100.0f * correct / (i + 1)) << "%" << std::endl;
        }
    }
    
    // End Timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    gpuExecutionTime = elapsed.count();   // in seconds
    
    float accuracy = 100.0f * correct / numTestImages;
    std::cout << "KNN Batched: Final accuracy: " << accuracy << "%" << std::endl;
    std::cout << "KNN Batched: GPU execution time: " << gpuExecutionTime << " seconds" << std::endl;
    std::cout << "KNN Batched: GPU memory usage: " << gpuMemoryUsage << " MB" << std::endl;
    
    // Clean up
    delete[] h_allDistances;
    delete[] h_kIndices;
    delete[] h_predictions;
    
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

int KNNClassifier::getBlockSize() const {
	return blockSize;
}