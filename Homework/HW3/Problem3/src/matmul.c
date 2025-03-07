#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N     512
#define LOOPS 10

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main() {
	double a[N][N]; /* input matrix */
	double b[N][N]; /* input matrix */
	double c[N][N]; /* result matrix */
	int    i, j, k, l, num_zeros;
	double start, finish, total;

	/* initialize a dense matrix */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = (double) (i + j);
			b[i][j] = (double) (i - j);
		}
	}

	printf("Starting dense matrix multiply... \n");
	start = CLOCK();

	#pragma omp parallel for private(i, j, k)
	for (l = 0; l < LOOPS; l++) {
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				c[i][j] = 0.0;
				for (k = 0; k < N; k++)
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
	}

	finish = CLOCK();

	total = finish - start;
	printf("Dense Matrix Multiplication Result: %g \n", c[7][8]); /* prevent dead code elimination */
	printf("Dense Matrix Multiplication Duration: %f ms\n", total);

	/* initialize a sparse matrix */
	num_zeros = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
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

	printf("Starting sparse matrix multiply... \n");
	start = CLOCK();
	for (l = 0; l < LOOPS; l++) {
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				c[i][j] = 0.0;
				for (k = 0; k < N; k++)
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
	}
	finish = CLOCK();
	total  = finish - start;
	printf("Sparse Matrix Multiplication Result: %g \n", c[7][8]); /* prevent dead code elimination */
	printf("Sparse Matrix Multiplication Duration: %f ms\n", total);
	printf("Sparse Matrix Multiplication Sparsity: %f \n", (float) num_zeros / (float) (N * N));

	return 0;
}
