// Program to compute PI using approximation based on Leibniz's formula and OpenMP
// Leibniz's formula: PI = 4 * (1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + ...)
// Author: Zander Ingare

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>

// Number of terms in the infinite series to use in the approximation
#define DEPTH 1000000

// Number of threads to use in the OpenMP implementation
#define NUM_THREADS 1

int main() {
	omp_set_num_threads(NUM_THREADS);

	// Start the timer
	auto start = std::chrono::high_resolution_clock::now();

	float sum = 0.0;

	#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < DEPTH; i++) {
		if (i % 2 == 0) {
			sum += 1.0 / (2 * i + 1);
		} else {
			sum -= 1.0 / (2 * i + 1);
		}
	}

	float approximated_pi = 4 * sum;

	// Stop the timer
	auto                            end          = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > elapsed_time = end - start;

	std::cout << "Approximated value of PI: " << std::setprecision(20) << approximated_pi
	          << std::endl;
	std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

	return 0;
}