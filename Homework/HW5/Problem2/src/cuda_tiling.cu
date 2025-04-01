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
#define TILE_SIZE  10
#define TILE_DIM   (TILE_SIZE + 2)   // Dimension of shared memory tile including halo (tile edges)
#define BLOCK_SIZE dim3(TILE_SIZE, TILE_SIZE, TILE_SIZE)
#define GRID_SIZE                         \
	dim3((N + TILE_SIZE - 1) / TILE_SIZE, \
	     (N + TILE_SIZE - 1) / TILE_SIZE, \
	     (N + TILE_SIZE - 1) / TILE_SIZE)

// Helper function for linear index in a 3D array
__host__ __device__ inline int idx3d(int i, int j, int k, int dimN) {
	return i * dimN * dimN + j * dimN + k;
}

void stencil_default(float* a, const float* b) {
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			for (int k = 1; k < N - 1; k++) {
				a[idx3d(i, j, k, N)] =
				    0.75f * (b[idx3d(i - 1, j, k, N)] + b[idx3d(i + 1, j, k, N)] +
				             b[idx3d(i, j - 1, k, N)] + b[idx3d(i, j + 1, k, N)] +
				             b[idx3d(i, j, k - 1, N)] + b[idx3d(i, j, k + 1, N)]);
			}
		}
	}
}

__global__ void stencil_kernel(float* a, const float* b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
		a[idx3d(i, j, k, N)] = 0.75f * (b[idx3d(i - 1, j, k, N)] + b[idx3d(i + 1, j, k, N)] +
		                                b[idx3d(i, j - 1, k, N)] + b[idx3d(i, j + 1, k, N)] +
		                                b[idx3d(i, j, k - 1, N)] + b[idx3d(i, j, k + 1, N)]);
	}
}

__global__ void stencil_kernel_tiled(float* a, const float* b) {
	// Shared memory tile including halo region
	__shared__ float tile[TILE_DIM][TILE_DIM][TILE_DIM];

	// Thread indices within the block
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	// Global indices corresponding to the start of the block
	int block_base_i = blockIdx.x * TILE_SIZE;
	int block_base_j = blockIdx.y * TILE_SIZE;
	int block_base_k = blockIdx.z * TILE_SIZE;

	// Global indices corresponding to this thread's primary element
	// This thread will be responsible for loading element (i,j,k) into tile(tx+1, ty+1, tz+1)
	int i = block_base_i + tx;
	int j = block_base_j + ty;
	int k = block_base_k + tz;

	// Load the central element this thread is responsible for from global memory into shared memory
	if (i < N && j < N && k < N) {
		tile[tx + 1][ty + 1][tz + 1] = b[idx3d(i, j, k, N)];
	}

	// Load -1 face halo elements from global memory to shared memory
	if (tx == 0 && i > 0)
		tile[0][ty + 1][tz + 1] = b[idx3d(i - 1, j, k, N)];
	if (ty == 0 && j > 0)
		tile[tx + 1][0][tz + 1] = b[idx3d(i, j - 1, k, N)];
	if (tz == 0 && k > 0)
		tile[tx + 1][ty + 1][0] = b[idx3d(i, j, k - 1, N)];

	// Load +1 face halo elements from global memory to shared memory
	if (tx == TILE_SIZE - 1 && i < N - 1)
		tile[TILE_SIZE + 1][ty + 1][tz + 1] = b[idx3d(i + 1, j, k, N)];
	if (ty == TILE_SIZE - 1 && j < N - 1)
		tile[tx + 1][TILE_SIZE + 1][tz + 1] = b[idx3d(i, j + 1, k, N)];
	if (tz == TILE_SIZE - 1 && k < N - 1)
		tile[tx + 1][ty + 1][TILE_SIZE + 1] = b[idx3d(i, j, k + 1, N)];

	__syncthreads();

	// Perform stencil computation using shared memory
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
		if (tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE) {
			// Access neighbors relative to the element at [tx+1][ty+1][tz+1]
			float neighbor_im1 = tile[tx][ty + 1][tz + 1];       // i-1 neighbor stored at tx
			float neighbor_ip1 = tile[tx + 2][ty + 1][tz + 1];   // i+1 neighbor stored at tx+2
			float neighbor_jm1 = tile[tx + 1][ty][tz + 1];       // j-1 neighbor stored at ty
			float neighbor_jp1 = tile[tx + 1][ty + 2][tz + 1];   // j+1 neighbor stored at ty+2
			float neighbor_km1 = tile[tx + 1][ty + 1][tz];       // k-1 neighbor stored at tz
			float neighbor_kp1 = tile[tx + 1][ty + 1][tz + 2];   // k+1 neighbor stored at tz+2

			a[idx3d(i, j, k, N)] = 0.75f * (neighbor_im1 + neighbor_ip1 + neighbor_jm1 +
			                                neighbor_jp1 + neighbor_km1 + neighbor_kp1);
		}
	}
}

int main() {
	const int size = N * N * N;

	// Host allocation
	float* h_a              = new float[size];
	float* h_b              = new float[size];
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
	std::cout << "Execution Time: " << std::fixed << std::setprecision(10) << elapsed.count()
	          << " seconds\n\n";

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
	bool  success   = true;
	float max_error = 0.0f;

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 1; j < N - 1; ++j) {
			for (int k = 1; k < N - 1; ++k) {
				int   idx   = idx3d(i, j, k, N);
				float error = fabs(h_a[idx] - h_a_ground_truth[idx]);
				if (error > 1e-5f) {
					std::cout << "Mismatch at (" << i << "," << j << "," << k << "): "
					          << "CPU = " << h_a_ground_truth[idx] << ", GPU = " << h_a[idx]
					          << ", error = " << error << "\n";
					success = false;
					break;
				}
				if (error > max_error)
					max_error = error;
			}
			if (!success)
				break;
		}
		if (!success)
			break;
	}

	if (success) {
		std::cout << "Results Match Ground-Truth!\n\n";
	} else {
		std::cout << "Results DO NOT Match Ground-Truth (Max Error: " << max_error << ")\n\n";
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
	success   = true;
	max_error = 0.0f;

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 1; j < N - 1; ++j) {
			for (int k = 1; k < N - 1; ++k) {
				int   idx   = idx3d(i, j, k, N);
				float error = fabs(h_a[idx] - h_a_ground_truth[idx]);
				if (error > 1e-5f) {
					std::cout << "Mismatch at (" << i << "," << j << "," << k << "): "
					          << "CPU = " << h_a_ground_truth[idx] << ", GPU = " << h_a[idx]
					          << ", error = " << error << "\n";
					success = false;
					break;
				}
				if (error > max_error)
					max_error = error;
			}
			if (!success)
				break;
		}
		if (!success)
			break;
	}

	if (success) {
		std::cout << "Results Match Ground-Truth!\n\n";
	} else {
		std::cout << "Results DO NOT Match Ground-Truth (Max Error: " << max_error << ")\n\n";
	}

	// Cleanup memory
	cudaFree(d_a);
	cudaFree(d_b);
	delete[] h_a;
	delete[] h_b;
	delete[] h_a_ground_truth;

	return 0;
}
