// A program that performs parallel histogramming of a data set using C++ and OpenMP.
// Converted from OpenMPI version by Zander Ingare

#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <chrono>

#define N        (1 << 23)
#define RANGE    100000
#define NUM_BINS 128
#define NUM_THREADS 1

int main() {
	omp_set_num_threads(NUM_THREADS);

    int* data = new int[N];
    for (int i = 0; i < N; i++) {
        data[i] = rand() % RANGE + 1;
    }

    float bin_width = static_cast<float>(RANGE) / NUM_BINS;

    int global_histogram[NUM_BINS] = {0};
	int local_histograms[NUM_THREADS][NUM_BINS] = {0};

	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		#pragma omp for
		for (int i = 0; i < N; i++) {
			int bin = floorf((data[i] - 1) / bin_width);
			if (bin >= NUM_BINS) bin = NUM_BINS - 1;
			local_histograms[tid][bin]++;
		}
	}

	// Combine local histograms into global histogram
	for (int t = 0; t < NUM_THREADS; ++t) {
		// #pragma omp parallel for
		for (int b = 0; b < NUM_BINS; ++b) {
			global_histogram[b] += local_histograms[t][b];
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "OpenMP Histogram Results\n";
    std::cout << "============================\n\n";
    std::cout << "Total elements: " << N << " | Range: 1-" << RANGE
              << " | Number of Bins: " << NUM_BINS << "\n";
    std::cout << "OpenMP Threads: " << NUM_THREADS << "\n\n";
    std::cout << "Performance Metrics:\n";
    std::cout << "  └── Total Time: " << std::fixed << std::setprecision(10)
              << elapsed.count() << " seconds\n\n";

    // for (int i = 0; i < NUM_BINS; i++) {
    //     int bin_start = (i == 0) ? 1 : floorf(i * bin_width) + 1;
    //     int bin_end = floor((i + 1) * bin_width);
    //     if (i == NUM_BINS - 1) bin_end = RANGE;

    //     std::cout << "Bin " << i << ": [" << bin_start << " - " << bin_end << "]\n";
    //     std::cout << "  └── Count: " << global_histogram[i] << "\n";
    // }

    delete[] data;
    return 0;
}
