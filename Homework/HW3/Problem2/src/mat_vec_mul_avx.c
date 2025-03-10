// Program to multiple a matrix and a vector using AVX-512F and without AVX-512F
// Author: Zander Ingare

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define MATRIX_ROWS 2048
#define MATRIX_COLS 2048

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void matrix_vector_multiplication(float mat[MATRIX_ROWS][MATRIX_COLS], float *vec, float *res) {
	for (int i = 0; i < MATRIX_ROWS; i++) {
		res[i] = 0.0;
		for (int j = 0; j < MATRIX_COLS; j++) {
			res[i] += mat[i][j] * vec[j];
		}
	}
}

void matrix_vector_multiplication_avx512f(float  mat[MATRIX_ROWS][MATRIX_COLS],
                                          float *vec,
                                          float *res) {
	for (int i = 0; i < MATRIX_ROWS; i++) {
		__m512 sum_vec = _mm512_setzero_ps();
		int    j       = 0;

		// Process 16 elements at a time
		for (; j + 16 <= MATRIX_COLS; j += 16) {
			__m512 mat_vec =
			    _mm512_loadu_ps(&mat[i][j]);   // Load 16 matrix elements into 512-bit AVX register
			__m512 vec_vec =
			    _mm512_loadu_ps(&vec[j]);   // Load 16 vector elements into 512-bit AVX register
			sum_vec = _mm512_fmadd_ps(mat_vec, vec_vec, sum_vec);   // Fused Multiply Add
		}

		// Store result of the vectorized sum
		float sum_arr[16];
		_mm512_storeu_ps(sum_arr, sum_vec);
		float sum = 0.0f;

		// Accumulate partial sums
		for (int k = 0; k < 16; k++) {
			sum += sum_arr[k];
		}

		// Handle remaining elements (if MATRIX_COLS is not a multiple of 16)
		for (; j < MATRIX_COLS; j++) {
			sum += mat[i][j] * vec[j];
		}

		res[i] = sum;   // Store the final sum for row i
	}
}

int main() {
	double start, finish, total, total_avx;
	int    i;
	float  matrix[MATRIX_ROWS][MATRIX_COLS];
	float  vector[MATRIX_COLS];
	float  result[MATRIX_ROWS];
	float  result_avx[MATRIX_ROWS];

	for (i = 0; i < MATRIX_ROWS; i++) {
		for (int j = 0; j < MATRIX_COLS; j++) {
			matrix[i][j] = (float) (i * MATRIX_ROWS + j);
			// matrix[i][j] = 1.0;
		}
	}

	for (i = 0; i < MATRIX_COLS; i++) {
		vector[i] = (float) (i);
		// vector[i] = 1.0;
	}

	start = CLOCK();
	matrix_vector_multiplication(matrix, vector, result);
	finish = CLOCK();
	total  = finish - start;

	printf("Matrix-Vector Multiplication Duration Without AVX: %f ms\n", total);

	start = CLOCK();
	matrix_vector_multiplication_avx512f(matrix, vector, result_avx);
	finish    = CLOCK();
	total_avx = finish - start;

	printf("Matrix-Vector Multiplication Duration With AVX: %f ms\n", total_avx);

	printf("AVX Speedup: %f\n", total / total_avx);

	for (i = 0; i < MATRIX_ROWS; i++) {
		if (result[i] != result_avx[i]) {
			printf("Results Do Not Match: \t");
			printf("result[%d] = %f, result_avx[%d] = %f\n", i, result[i], i, result_avx[i]);
			return 0;
		}
	}
	printf("Results Match\n");
	return 0;
}
