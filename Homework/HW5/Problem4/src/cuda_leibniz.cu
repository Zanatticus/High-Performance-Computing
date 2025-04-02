/*
Approximation of the number PI through the Leibniz's series using CUDA

Based on the following sequential algorithm:

    double s  = 1;   // Signal for the next iteration
    double pi = 0;

    for (double i = 1; i <= (N * 2); i += 2) {
        pi = pi + s * (4 / i);
        s  = -s;
    }

Author: Zander Ingare
*/

#include <chrono>
#include <cmath>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define N          (1 << 29)   // Number of iterations
#define BLOCK_SIZE 1024
#define GRID_SIZE  (N + BLOCK_SIZE - 1) / BLOCK_SIZE

float compute_pi_float(int n) {
	float s  = 1.0f;
	float pi = 0.0f;

	for (int i = 1; i <= n * 2; i += 2) {
		pi += s * (4.0f / i);
		s = -s;
	}

	return pi;
}

double compute_pi_double(int n) {
	double s  = 1.0;
	double pi = 0.0;

	for (int i = 1; i <= n * 2; i += 2) {
		pi += s * (4.0 / i);
		s = -s;
	}

	return pi;
}

__global__ void leibniz_kernel(float* result) {
	__shared__ float sdata[BLOCK_SIZE];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		float sign         = (tid % 2 == 0) ? 1.0f : -1.0f;
		float denom        = 2.0f * tid + 1.0f;
		sdata[threadIdx.x] = sign * (4.0f / denom);
	} else {
		sdata[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	// In-place reduction in shared memory
	for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Thread 0 writes the final result for the block
	if (threadIdx.x == 0) {
		atomicAdd(result, sdata[0]);
	}
}

__global__ void leibniz_kernel(double* result) {
	__shared__ double sdata[BLOCK_SIZE];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		double sign        = (tid % 2 == 0) ? 1.0 : -1.0;
		double denom       = 2.0 * tid + 1.0;
		sdata[threadIdx.x] = sign * (4.0 / denom);
	} else {
		sdata[threadIdx.x] = 0.0;
	}
	__syncthreads();

	// In-place reduction in shared memory
	for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Thread 0 writes the final result for the block
	if (threadIdx.x == 0) {
		atomicAdd(result, sdata[0]);
	}
}

int main() {
	printf("Approximation of the number PI through the Leibniz's series\n");

	// Execute the Leibniz series on CPU using single precision
	auto                          float_start   = std::chrono::high_resolution_clock::now();
	float                         pi_float      = compute_pi_float(N);
	auto                          float_end     = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> float_elapsed = float_end - float_start;

	std::cout << "==========================================\n";
	std::cout << " CPU Leibniz Float Pi Computation Results\n";
	std::cout << "==========================================\n\n";
	std::cout << "Total iterations: " << N << "\n";
	std::cout << "Execution Time (CPU): " << std::fixed << std::setprecision(10)
	          << float_elapsed.count() << " seconds\n";
	std::cout << "Approximation of PI: " << std::setprecision(15) << pi_float << "\n\n";

	// Execute the Leibniz series on CPU using double precision
	auto                          double_start   = std::chrono::high_resolution_clock::now();
	double                        pi_double      = compute_pi_double(N);
	auto                          double_end     = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> double_elapsed = double_end - double_start;

	std::cout << "===========================================\n";
	std::cout << " CPU Leibniz Double Pi Computation Results\n";
	std::cout << "===========================================\n\n";
	std::cout << "Total iterations: " << N << "\n";
	std::cout << "Execution Time (CPU): " << std::fixed << std::setprecision(10)
	          << double_elapsed.count() << " seconds\n";
	std::cout << "Approximation of PI: " << std::setprecision(15) << pi_double << "\n\n";

	// Allocate memory on host
	float*  h_result_float  = new float;
	double* h_result_double = new double;

	// Allocate memory on device
	float*  d_result_float;
	double* d_result_double;
	cudaMalloc(&d_result_float, sizeof(float));
	cudaMalloc(&d_result_double, sizeof(double));

	// Launch single precision kernel
	float_start = std::chrono::high_resolution_clock::now();

	leibniz_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_float);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_float, d_result_float, sizeof(float), cudaMemcpyDeviceToHost);

	float_end     = std::chrono::high_resolution_clock::now();
	float_elapsed = float_end - float_start;

	// Print performance results
	std::cout << "===========================================\n";
	std::cout << " CUDA Leibniz Float Pi Computation Results\n";
	std::cout << "===========================================\n\n";
	std::cout << "Total iterations: " << N << "\n";
	std::cout << "Block Size (Threads Per Block): " << BLOCK_SIZE
	          << " | Grid Size (Number of Blocks): " << GRID_SIZE << "\n";
	std::cout << "Execution Time (including device sync & copy): " << std::fixed
	          << std::setprecision(10) << float_elapsed.count() << " seconds\n";
	std::cout << "Approximation of PI: " << std::setprecision(15) << *h_result_float << "\n\n";

	// Launch double precision kernel
	double_start = std::chrono::high_resolution_clock::now();

	leibniz_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_double);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_double, d_result_double, sizeof(double), cudaMemcpyDeviceToHost);

	double_end     = std::chrono::high_resolution_clock::now();
	double_elapsed = double_end - double_start;

	// Print performance results
	std::cout << "============================================\n";
	std::cout << " CUDA Leibniz Double Pi Computation Results\n";
	std::cout << "============================================\n\n";
	std::cout << "Total iterations: " << N << "\n";
	std::cout << "Block Size (Threads Per Block): " << BLOCK_SIZE
	          << " | Grid Size (Number of Blocks): " << GRID_SIZE << "\n";
	std::cout << "Execution Time (including device sync & copy): " << std::fixed
	          << std::setprecision(10) << double_elapsed.count() << " seconds\n";
	std::cout << "Approximation of PI: " << std::setprecision(15) << *h_result_double << "\n\n";

	// Cleanup memory
	cudaFree(d_result_float);
	cudaFree(d_result_double);
	free(h_result_float);
	free(h_result_double);
}
