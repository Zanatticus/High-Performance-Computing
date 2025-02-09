// Program to compute PI using approximation based on Leibniz's formula and
// Pthreads Leibniz's formula: PI = 4 * (1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + ...)
// Author: Zander Ingare

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <vector>

// Number of terms in the infinite series to use in the approximation
#define DEPTH 1000000

// Number of threads to use in the computation
#define MAX_THREADS 4

// Array to store the sum of the series computed by each thread
float thread_sums[MAX_THREADS] = {0};

void* compute_pi(void* arg) {
	int thread_id = *(int*) arg;
	int start     = thread_id * DEPTH / MAX_THREADS;
	int end       = (thread_id + 1) * DEPTH / MAX_THREADS;

	for (int i = start; i < end; i++) {
		if (i % 2 == 0) {
			thread_sums[thread_id] += 1.0 / (2 * i + 1);
		} else {
			thread_sums[thread_id] -= 1.0 / (2 * i + 1);
		}

		// This line showcases the inaccuracies of floating point arithmetic in
		// programming (use a DEPTH value of 5 to see the difference) std::cout
		// << std::setprecision(20) << "SUM: " <<thread_sums[thread_id] <<
		// std::endl;
	}
	return NULL;
}

int main() {
	pthread_t threads[MAX_THREADS];
	int       thread_ids[MAX_THREADS];

	// Start the timer
	auto start = std::chrono::high_resolution_clock::now();

	// Create threads to approximate PI
	for (int i = 0; i < MAX_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, compute_pi, &thread_ids[i]);
	}

	// Join threads after all threads complete
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	// Add the sum of all thread parts to get the final approximation
	float approximated_pi = 0.0;
	for (int i = 0; i < MAX_THREADS; i++) {
		approximated_pi += thread_sums[i];
	}
	approximated_pi *= 4;

	// Stop the timer
	auto                            end          = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > elapsed_time = end - start;

	std::cout << "Approximated value of PI: " << std::setprecision(20) << approximated_pi
	          << std::endl;
	std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
}