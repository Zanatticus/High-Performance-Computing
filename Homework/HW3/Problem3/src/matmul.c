#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N     512
#define LOOPS 10

typedef struct {
	double *values;
	int    *column_indices;
	int    *row_pointers;
	int     num_nonzeros;
	int     matrix_size;
} CSRMatrix;

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void initialize_dense_matrices(double a[N][N], double b[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = (double) (i + j);
			b[i][j] = (double) (i - j);
		}
	}
}

int initialize_sparse_matrices(double a[N][N], double b[N][N]) {
	int num_zeros = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if ((i < j) && (i % 2 > 0)) {
				a[i][j] = (double) (i + j);
				b[i][j] = (double) (i - j);
			} else {
				num_zeros++;
				a[i][j] = 0.0;
				b[i][j] = 0.0;
			}
		}
	}
	return num_zeros;
}

void convert_to_csr(double a[N][N], CSRMatrix *csr) {
	int nnz = 0;

	// Count the number of nonzero elements
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (a[i][j] != 0.0) {
				nnz++;
			}
		}
	}

	// Allocate memory for CSR arrays
	csr->values         = (double *) malloc(nnz * sizeof(double));
	csr->column_indices = (int *) malloc(nnz * sizeof(int));
	csr->row_pointers   = (int *) malloc((N + 1) * sizeof(int));
	csr->num_nonzeros   = nnz;
	csr->matrix_size    = N;

	// Fill CSR arrays
	int index            = 0;
	csr->row_pointers[0] = 0;   // First row starts at index 0
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (a[i][j] != 0.0) {
				csr->values[index]         = a[i][j];
				csr->column_indices[index] = j;
				index++;
			}
		}
		csr->row_pointers[i + 1] = index;   // Mark start of next row
	}
}

void convert_from_csr(CSRMatrix *csr, double a[N][N]) {
	// Initialize all elements to zero
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = 0.0;
		}
	}

	// Populate matrix from CSR format
	for (int i = 0; i < csr->matrix_size; i++) {
		for (int idx = csr->row_pointers[i]; idx < csr->row_pointers[i + 1]; idx++) {
			int col   = csr->column_indices[idx];
			a[i][col] = csr->values[idx];
		}
	}
}

void free_csr(CSRMatrix *csr) {
	free(csr->values);
	free(csr->column_indices);
	free(csr->row_pointers);
}

// Matrix multiplication for dense matrices
void matrix_multiply(double a[N][N], double b[N][N], double c[N][N]) {
	for (int l = 0; l < LOOPS; l++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				c[i][j] = 0.0;
				for (int k = 0; k < N; k++)
					c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void matrix_multiply_csr(CSRMatrix *a_csr, CSRMatrix *b_csr, CSRMatrix *c_csr) {
	// Allocate maximum possible space
	c_csr->values = (double *) malloc(a_csr->num_nonzeros * b_csr->matrix_size * sizeof(double));
	c_csr->column_indices = (int *) malloc(a_csr->num_nonzeros * b_csr->matrix_size * sizeof(int));
	c_csr->row_pointers   = (int *) malloc((a_csr->matrix_size + 1) * sizeof(int));
	c_csr->num_nonzeros   = 0;
	c_csr->matrix_size    = a_csr->matrix_size;

	int nnz                = 0;
	c_csr->row_pointers[0] = 0;

	// Iterate over each row of A
	for (int i = 0; i < a_csr->matrix_size; i++) {
		double temp_result[N] = {0};

		// Iterate over nonzero elements in row i of A
		for (int a_idx = a_csr->row_pointers[i]; a_idx < a_csr->row_pointers[i + 1]; a_idx++) {
			int    k   = a_csr->column_indices[a_idx];   // Column index (row index in B)
			double Aik = a_csr->values[a_idx];           // Value of A[i, k]

			// Iterate over nonzero elements in row k of B
			for (int b_idx = b_csr->row_pointers[k]; b_idx < b_csr->row_pointers[k + 1]; b_idx++) {
				int    j   = b_csr->column_indices[b_idx];   // Column index in B (column in result)
				double Bkj = b_csr->values[b_idx];           // Value of B[k, j]

				// Accumulate multiplication result into temporary storage
				temp_result[j] += Aik * Bkj;
			}
		}

		// Store nonzero results into CSR structure for C
		for (int j = 0; j < a_csr->matrix_size; j++) {
			if (temp_result[j] != 0.0) {
				c_csr->values[nnz]         = temp_result[j];
				c_csr->column_indices[nnz] = j;
				nnz++;
			}
		}
		c_csr->row_pointers[i + 1] = nnz;   // Mark start of next row
	}

	c_csr->num_nonzeros = nnz;   // Update nonzero count

	// Reallocate memory to fit exact size
	c_csr->values         = (double *) realloc(c_csr->values, nnz * sizeof(double));
	c_csr->column_indices = (int *) realloc(c_csr->column_indices, nnz * sizeof(int));
}

int main() {
	double    a[N][N]; /* input matrix */
	double    b[N][N]; /* input matrix */
	double    c[N][N]; /* result matrix */
	double    d[N][N]; /* result matrix */
	int       num_zeros;
	double    start, finish;
	CSRMatrix a_csr, b_csr, c_csr;

	/*---------------------------------------------------------------------------------------------*/
	initialize_dense_matrices(a, b);

	start = CLOCK();
	matrix_multiply(a, b, c);
	finish = CLOCK();

	printf("Dense Matrix Multiplication Result: %g \n", c[7][8]);   // Avoids dead code elimination
	printf("Dense Matrix Multiplication Duration: %f ms\n", finish - start);
	/*---------------------------------------------------------------------------------------------*/

	/*---------------------------------------------------------------------------------------------*/
	num_zeros = initialize_sparse_matrices(a, b);

	convert_to_csr(a, &a_csr);
	convert_to_csr(b, &b_csr);

	start = CLOCK();
	matrix_multiply_csr(&a_csr, &b_csr, &c_csr);
	finish = CLOCK();

	convert_from_csr(&c_csr, c);

	matrix_multiply(a, b, d);

	// Check if the results are the same
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (c[i][j] != d[i][j]) {
				printf("Error: Result mismatch at index (%d, %d)\n", i, j);
				return 1;
			}
		}
	}

	printf("Sparse Matrix Multiplication Result: %g \n", c[7][8]);   // Avoids dead code elimination
	printf("Sparse Matrix Multiplication Duration: %f ms\n", finish - start);
	printf("Sparse Matrix Multiplication Sparsity: %f \n", (float) num_zeros / (float) (N * N));
	/*---------------------------------------------------------------------------------------------*/

	free_csr(&a_csr);
	free_csr(&b_csr);
	free_csr(&c_csr);

	return 0;
}
