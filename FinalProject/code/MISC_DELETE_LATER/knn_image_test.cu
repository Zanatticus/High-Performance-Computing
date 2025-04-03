/*
A program to test the KNN image classification algorithm on a set of images.
This program uses the KNN algorithm to classify images based on their pixel values.

1. Computes the Euclidean distance to every image in the training set
2. Finds the nearest neighbor (K=1) with the smallest distance from the test image
3. Classifies the test image based on the label of the nearest neighbor

The test set feeds each image to the CUDA kernel and gets a vector returned with the predicted label for each image.

Author: Zander Ingare
*/

#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#define DATASET_SIZE 60000
#define IMAGE_SIZE 784 // 28x28 images
#define K 1 // Number of nearest neighbors


// Reads a 32-bit big-endian integer from file
int readBigEndianInt(FILE *fp) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fp);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Load MNIST image file into a float array normalized to [0.0, 1.0]
float *loadMNISTImages(const char *filename, int *number_of_images_out) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int magic = readBigEndianInt(fp);
    if (magic != 2051) {
        fprintf(stderr, "Invalid MNIST image file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int num_images = readBigEndianInt(fp);
    int num_rows = readBigEndianInt(fp);
    int num_cols = readBigEndianInt(fp);
    int image_size = num_rows * num_cols;

    unsigned char *buffer = (unsigned char *)malloc(num_images * image_size);
    fread(buffer, sizeof(unsigned char), num_images * image_size, fp);
    fclose(fp);

    float *images = (float *)malloc(num_images * image_size * sizeof(float));
    for (int i = 0; i < num_images * image_size; i++) {
        images[i] = buffer[i] / 255.0f;  // normalize to [0.0, 1.0]
    }

    free(buffer);
    *number_of_images_out = num_images;
    return images;
}
















// CUDA kernel to compute Euclidean distances between test image and all training images
__global__ void computeEuclideanDistances(const float *trainingImages, const float *testImage, float *distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < DATASET_SIZE) {
        float distance = 0.0f;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            float diff = trainingImages[idx * IMAGE_SIZE + i] - testImage[i];
            distance += diff * diff;
        }
        distances[idx] = sqrtf(distance);
    }
}

// Structure to store distance-label pairs
typedef struct {
    float distance;
    unsigned char label;
} DistanceLabelPair;

// Comparison function for sorting
int comparePairs(const void *a, const void *b) {
    DistanceLabelPair *pairA = (DistanceLabelPair *)a;
    DistanceLabelPair *pairB = (DistanceLabelPair *)b;
    return (pairA->distance > pairB->distance) - (pairA->distance < pairB->distance);
}

// Perform majority voting on top-K labels
unsigned char majorityVote(DistanceLabelPair *topK) {
    int labelCount[256] = {0}; // assuming label is 0–255

    for (int i = 0; i < K; i++) {
        labelCount[topK[i].label]++;
    }

    int maxCount = 0;
    unsigned char majorityLabel = 0;
    for (int i = 0; i < 256; i++) {
        if (labelCount[i] > maxCount) {
            maxCount = labelCount[i];
            majorityLabel = i;
        }
    }

    return majorityLabel;
}

int main() {
    // Host memory allocations
    float *h_training_images = (float *)malloc(DATASET_SIZE * IMAGE_SIZE * sizeof(float));
    float *h_test_image = (float *)malloc(IMAGE_SIZE * sizeof(float));
    unsigned char *h_training_labels = (unsigned char *)malloc(DATASET_SIZE * sizeof(unsigned char));

    // Fill with real data in practice — here zero-filled as placeholders
    for (int i = 0; i < DATASET_SIZE * IMAGE_SIZE; i++) h_training_images[i] = 0.0f;
    for (int i = 0; i < IMAGE_SIZE; i++) h_test_image[i] = 0.0f;
    for (int i = 0; i < DATASET_SIZE; i++) h_training_labels[i] = i % 10; // dummy 0–9 labels

    // Device memory allocations
    float *d_training_images, *d_test_image, *d_distances;
    cudaMalloc(&d_training_images, DATASET_SIZE * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_test_image, IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_distances, DATASET_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_training_images, h_training_images, DATASET_SIZE * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_image, h_test_image, IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (DATASET_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    computeEuclideanDistances<<<blocksPerGrid, threadsPerBlock>>>(d_training_images, d_test_image, d_distances);
    cudaDeviceSynchronize();

    // Copy distances back to host
    float *h_distances = (float *)malloc(DATASET_SIZE * sizeof(float));
    cudaMemcpy(h_distances, d_distances, DATASET_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Pair distances with labels
    DistanceLabelPair *pairs = (DistanceLabelPair *)malloc(DATASET_SIZE * sizeof(DistanceLabelPair));
    for (int i = 0; i < DATASET_SIZE; i++) {
        pairs[i].distance = h_distances[i];
        pairs[i].label = h_training_labels[i];
    }

    // Sort to find top-K
    qsort(pairs, DATASET_SIZE, sizeof(DistanceLabelPair), comparePairs);

    // Get predicted label from majority of top-K
    unsigned char predictedLabel = majorityVote(pairs);

    // Output results
    printf("Predicted label (K=%d): %d\n", K, predictedLabel);

    // Cleanup
    free(h_training_images);
    free(h_test_image);
    free(h_training_labels);
    free(h_distances);
    free(pairs);
    cudaFree(d_training_images);
    cudaFree(d_test_image);
    cudaFree(d_distances);

    return 0;
}