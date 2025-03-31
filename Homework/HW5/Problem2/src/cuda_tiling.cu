/*
A program that optimizes a "nearest neighbor" or "stencil" computation using CUDA on a single GPU.

Based on the example below:
    #define n 32
    float a[n][n][n], b[n][n][n];
    for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
            for (k = 1; k < n - 1; k++) {
                a[i][j][k] = 0.75 * (b[i - 1][j][k] + b[i + 1][j][k] + b[i][j - 1][k] +
                                        b[i][j + 1][k] + b[i][j][k - 1] + b[i][j][k + 1]);
            }

Author: Zander Ingare
*/

#include <chrono>
#include <cuda.h>
#include <iomanip>
#include <iostream>

#define N          32
#define TILE_SIZE  4
#define BLOCK_SIZE dim3(TILE_SIZE, TILE_SIZE, TILE_SIZE)
#define GRID_SIZE  dim3((N + TILE_SIZE - 1) / TILE_SIZE, \
						(N + TILE_SIZE - 1) / TILE_SIZE, \
	     			    (N + TILE_SIZE - 1) / TILE_SIZE)

void stencil_default(float* a, const float* b) {
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			for (int k = 1; k < N - 1; k++) {
				a[i * N * N + j * N + k] =
					0.75f * (b[(i - 1) * N * N + j * N + k] +
							 b[(i + 1) * N * N + j * N + k] +
							 b[i * N * N + (j - 1) * N + k] +
							 b[i * N * N + (j + 1) * N + k] +
							 b[i * N * N + j * N + (k - 1)] +
							 b[i * N * N + j * N + (k + 1)]);
			}
		}
	}
}
						
__global__ void stencil_kernel(float* a, const float* b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= N || j >= N || k >= N) {
		return;
	}

	a[i * N * N + j * N + k] =
	    0.75f * (b[(i - 1) * N * N + j * N + k] + b[(i + 1) * N * N + j * N + k] +
	             b[i * N * N + (j - 1) * N + k] + b[i * N * N + (j + 1) * N + k] +
	             b[i * N * N + j * N + (k - 1)] + b[i * N * N + j * N + (k + 1)]);
}

__global__ void stencil_kernel_tiled(float* a, const float* b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= N || j >= N || k >= N) {
		return;
	}

	// Create a tile halo for data access to +1 and -1 neighbors
	__shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];

	// Load data for each tile's corresponding central value from global memory into shared memory
	tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1] = b[i * N * N + j * N + k];

	// Load the halo (edge) values from global memory into shared memory
	if (i > 0)
		tile[threadIdx.x][threadIdx.y + 1][threadIdx.z + 1] = b[(i - 1) * N * N + j * N + k];
	if (i < N - 1)
		tile[threadIdx.x + 2][threadIdx.y + 1][threadIdx.z + 1] = b[(i + 1) * N * N + j * N + k];
	if (j > 0)
		tile[threadIdx.x + 1][threadIdx.y][threadIdx.z + 1] = b[i * N * N + (j - 1) * N + k];
	if (j < N - 1)
		tile[threadIdx.x + 1][threadIdx.y + 2][threadIdx.z + 1] = b[i * N * N + (j + 1) * N + k];
	if (k > 0)
		tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z] = b[i * N * N + j * N + (k - 1)];
	if (k < N - 1)
		tile[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 2] = b[i * N * N + j * N + (k + 1)];

	// Perform the stencil computation using shared memory
	if (i > 0 && i < N - 1 &&
		j > 0 && j < N - 1 &&
		k > 0 && k < N - 1) {
		a[i * N * N + j * N + k] = 0.75f * (tile[threadIdx.x - 1][threadIdx.y][threadIdx.z] +
		                                    tile[threadIdx.x + 1][threadIdx.y][threadIdx.z] +
		                                    tile[threadIdx.x][threadIdx.y - 1][threadIdx.z] +
		                                    tile[threadIdx.x][threadIdx.y + 1][threadIdx.z] +
		                                    tile[threadIdx.x][threadIdx.y][threadIdx.z - 1] +
		                                    tile[threadIdx.x][threadIdx.y][threadIdx.z + 1]);
	}

	__syncthreads();
}

int main() {
	const int size = N * N * N;

	// Host allocation
	float* h_a = new float[size];
	float* h_b = new float[size];
	float* h_a_ground_truth = new float[size];

	for (int i = 0; i < size; ++i) {
		h_b[i] = static_cast<float>(i);
	}

	// Compute the ground truth using the default CPU implementation
	auto start = std::chrono::high_resolution_clock::now();
	stencil_default(h_a_ground_truth, h_b);
	auto                          end     = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;

	// Print performance results
	std::cout << "==========================================\n";
	std::cout << "C/C++ Default Computation Results\n";
	std::cout << "==========================================\n\n";
	std::cout << "Total elements: " << N * N * N << "\n";
	std::cout << "Execution Time: " << std::fixed
	          << std::setprecision(10) << elapsed.count() << " seconds\n\n";

	// Device allocation
	float *d_a, *d_b;
	cudaMalloc(&d_a, size * sizeof(float));
	cudaMalloc(&d_b, size * sizeof(float));
	cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

	// Launch non-tiled kernel
	start = std::chrono::high_resolution_clock::now();
	stencil_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b);

	// Synchronize kernel with the host
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	end     = std::chrono::high_resolution_clock::now();
	elapsed = end - start;

	// Print performance results
	std::cout << "==========================================\n";
	std::cout << "CUDA Non-Tiled Stencil Computation Results\n";
	std::cout << "==========================================\n\n";
	std::cout << "Total elements: " << N * N * N << "\n";
	std::cout << "Block Size (Threads Per Block): (" << BLOCK_SIZE.x << ", " << BLOCK_SIZE.y << ", "
	          << BLOCK_SIZE.z << ") = " << BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z
	          << " | Grid Size (Number of Blocks): (" << GRID_SIZE.x << ", " << GRID_SIZE.y << ", "
	          << GRID_SIZE.z << ") = " << GRID_SIZE.x * GRID_SIZE.y * GRID_SIZE.z << "\n";
	std::cout << "Execution Time (including device sync & copy): " << std::fixed
	          << std::setprecision(10) << elapsed.count() << " seconds\n\n";

	// Verify results (non-tiled)
	std::cout << "Verifying Non-Tiled Kernel Results...\n";
	bool success = true;
	float max_error = 0.0f;

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 1; j < N - 1; ++j) {
			for (int k = 1; k < N - 1; ++k) {
				int idx = i * N * N + j * N + k;
				float error = fabs(h_a[idx] - h_a_ground_truth[idx]);
				if (error > 1e-5f) {
					std::cout << "Mismatch at (" << i << "," << j << "," << k << "): "
					          << "CPU = " << h_a_ground_truth[idx]
					          << ", GPU = " << h_a[idx]
					          << ", error = " << error << "\n";
					success = false;
				}
				if (error > max_error) max_error = error;
			}
		}
	}

	if (success) {
		std::cout << "Results Match Ground-Truth\n\n";
	} else {
		std::cout << "Results DO NOT Match Ground-Truth\n\n";
	}

	// Launch Tiled Kernel
	start = std::chrono::high_resolution_clock::now();
	stencil_kernel_tiled<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b);

	// Synchronize kernel with the host
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	end     = std::chrono::high_resolution_clock::now();
	elapsed = end - start;

	// Print performance results
	std::cout << "==========================================\n";
	std::cout << "CUDA Tiled Stencil Computation Results\n";
	std::cout << "==========================================\n\n";
	std::cout << "Total elements: " << N * N * N << "\n";
	std::cout << "Tile Size: " << TILE_SIZE << " | Block Size (Threads Per Block): ("
	          << BLOCK_SIZE.x << ", " << BLOCK_SIZE.y << ", " << BLOCK_SIZE.z
	          << ") = " << BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z
	          << " | Grid Size (Number of Blocks): (" << GRID_SIZE.x << ", " << GRID_SIZE.y << ", "
	          << GRID_SIZE.z << ") = " << GRID_SIZE.x * GRID_SIZE.y * GRID_SIZE.z << "\n";
	std::cout << "Execution Time (including device sync & copy): " << std::fixed
	          << std::setprecision(10) << elapsed.count() << " seconds\n\n";

	// Verify results (tiled)
	std::cout << "Verifying Tiled Kernel Results...\n";
	success = true;
	max_error = 0.0f;

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 1; j < N - 1; ++j) {
			for (int k = 1; k < N - 1; ++k) {
				int idx = i * N * N + j * N + k;
				float error = fabs(h_a[idx] - h_a_ground_truth[idx]);
				if (error > 1e-5f) {
					std::cout << "Mismatch at (" << i << "," << j << "," << k << "): "
					          << "CPU = " << h_a_ground_truth[idx]
					          << ", GPU = " << h_a[idx]
					          << ", error = " << error << "\n";
					success = false;
				}
				if (error > max_error) max_error = error;
			}
		}
	}

	if (success) {
		std::cout << "Results Match Ground-Truth\n\n";
	} else {
		std::cout << "Results DO NOT Match Ground-Truth\n\n";
	}

	// Cleanup memory
	cudaFree(d_a);
	cudaFree(d_b);
	delete[] h_a;
	delete[] h_b;
	delete[] h_a_ground_truth;

	return 0;
}
