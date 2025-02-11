// Program to compute PI using Monte Carlo simulation and OpenMP
// Author: Zander Ingare

#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>

// Number of points to use in the Monte Carlo simulation
#define NUM_POINTS 1000000

// Number of threads to use in the OpenMP implementation
#define NUM_THREADS 1

int main() {
	omp_set_num_threads(NUM_THREADS);

	// Start the timer
	auto start = std::chrono::high_resolution_clock::now();

	int points_in_circle = 0;

#pragma omp parallel
	{
		// Create a random number generator for each thread
		std::random_device               rd;
		std::mt19937                     gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp for reduction(+ : points_in_circle)
		for (int i = 0; i < NUM_POINTS; i++) {
			double x = dis(gen);
			double y = dis(gen);
			if (x * x + y * y <= 1.0) {
				points_in_circle++;
			}
		}
	}

	// Calculate the approximation of PI
	double approximated_pi = 4.0 * points_in_circle / NUM_POINTS;

	// Stop the timer
	auto                            end          = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > elapsed_time = end - start;

	std::cout << "Approximated value of PI: " << std::setprecision(20) << approximated_pi
	          << std::endl;
	std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

	return 0;
}