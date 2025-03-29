// A program that performs histogramming of a data set using CUDA on a single GPU.
// Author: Zander Ingare

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define N        (1 << 23)  // Can be varied from 2^12 to 2^23
#define RANGE    100000
#define NUM_BINS 128
#define BLOCK_SIZE 256
#define GRID_SIZE (N + BLOCK_SIZE - 1) / BLOCK_SIZE

__global__ void histogram_kernel(const int* data, int* histogram, int* example_values, int n, float bin_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int bin = floorf((data[idx] - 1) / bin_width);
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;

    atomicAdd(&histogram[bin], 1);

    if (atomicCAS(&example_values[bin], 0, data[idx]) == 0) {
        example_values[bin] = data[idx];
    }
}

int main() {
    int* h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % RANGE + 1;
    }

    float bin_width = static_cast<float>(RANGE) / NUM_BINS;

    int* d_data = nullptr;
    int* d_histogram = nullptr;
    int* d_example_values = nullptr;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(int));
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));
    cudaMalloc(&d_example_values, NUM_BINS * sizeof(int));
    cudaMemset(d_example_values, 0, NUM_BINS * sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    int* h_histogram = new int[NUM_BINS];
    int* h_example_values = new int[NUM_BINS];

    auto start = std::chrono::high_resolution_clock::now();
    histogram_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, d_histogram, d_example_values, N, bin_width);
    cudaDeviceSynchronize();
    cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_example_values, d_example_values, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "CUDA Histogram Results\n";
    std::cout << "============================\n\n";
    std::cout << "Total elements: " << N << " | Range: 1-" << RANGE
              << " | Number of Bins: " << NUM_BINS << "\n";
    std::cout << "Block Size (Threads Per Block): " << BLOCK_SIZE << " | Grid Size (Threads Per Block): " << GRID_SIZE << "\n";
    std::cout << "Execution Time (including device sync & copy): " 
              << std::fixed << std::setprecision(10) << elapsed.count() << " seconds\n\n";

    for (int i = 0; i < NUM_BINS; ++i) {
        int bin_start = (i == 0) ? 1 : static_cast<int>(i * bin_width) + 1;
        int bin_end   = static_cast<int>((i + 1) * bin_width);
        if (i == NUM_BINS - 1) bin_end = RANGE;

        std::cout << "Bin " << i << ": [" << bin_start << " - " << bin_end << "]\n";
        std::cout << "  └── Count: " << h_histogram[i] << "\n";
        std::cout << "  └── Example Value: " << h_example_values[i] << "\n";
    }

    delete[] h_data;
    delete[] h_histogram;
    delete[] h_example_values;
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_example_values);

    return 0;
}
