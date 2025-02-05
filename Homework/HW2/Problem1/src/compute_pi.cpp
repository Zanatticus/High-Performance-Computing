// Program to compute PI using approximation based on Leibniz's formula and Pthreads
// Leibniz's formula: PI = 4 * (1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + ...)
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
float thread_sums[MAX_THREADS];     

void* compute_pi(void* arg) {
    pthread_t thread_id = pthread_self();

    int start = thread_id * DEPTH / MAX_THREADS;
    int end = (thread_id + 1) * DEPTH / MAX_THREADS;

    for (int i = start; i < end; i++) {
        thread_sums[thread_id] += (1 - 1 / (2 * i + 1));
    }

    return NULL;
}

int main() {
    
    pthread_t threads[MAX_THREADS];

    // Create threads to approximate PI
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, compute_pi, (void*) NULL);
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

    std::cout << "Approximated value of PI: " << std::setprecision(15) << approximated_pi << std::endl;
}