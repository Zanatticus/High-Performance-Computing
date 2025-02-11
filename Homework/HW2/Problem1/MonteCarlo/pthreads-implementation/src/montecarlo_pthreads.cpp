// Program to compute PI using Monte Carlo simulation and Pthreads
// Author: Zander Ingare

#include <chrono>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <random>
#include <vector>

// Number of "darts" thrown in the simulation
#define NUM_POINTS 1000000

// Number of threads to use in the computation
#define NUM_THREADS 4

struct ThreadData {
	int points_in_circle;
	int points_per_thread;
};

void* compute_pi(void* arg) {
	ThreadData* data             = (ThreadData*) arg;
	int         points_in_circle = 0;

	// Create a random number generator with a Mersenne Twister 19937 generator engine
	std::random_device               rd;
	std::mt19937                     gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	for (int i = 0; i < data->points_per_thread; i++) {
		double x = dis(gen);
		double y = dis(gen);
		if (x * x + y * y <= 1.0) {
			points_in_circle++;
		}
	}

	data->points_in_circle = points_in_circle;
	pthread_exit(nullptr);
}

int main() {
	// Start the timer
	auto start = std::chrono::high_resolution_clock::now();

	pthread_t  threads[NUM_THREADS];
	ThreadData thread_data[NUM_THREADS];
	int        points_per_thread = NUM_POINTS / NUM_THREADS;

	for (int i = 0; i < NUM_THREADS; i++) {
		thread_data[i].points_per_thread = points_per_thread;
		pthread_create(&threads[i], nullptr, compute_pi, &thread_data[i]);
	}

	int total_points_in_circle = 0;
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], nullptr);
		total_points_in_circle += thread_data[i].points_in_circle;
	}

	// Calculate the approximation of PI
	double approximated_pi = 4.0 * total_points_in_circle / NUM_POINTS;

	// Stop the timer
	auto                            end          = std::chrono::high_resolution_clock::now();
	std::chrono::duration< double > elapsed_time = end - start;

	std::cout << "Approximated value of PI: " << std::setprecision(20) << approximated_pi
	          << std::endl;
	std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

	return 0;
}