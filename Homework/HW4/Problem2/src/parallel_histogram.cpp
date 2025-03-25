// A program that performs parallel histogramming of a data set using C++ and OpenMPI.
// Author: Zander Ingare

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <iomanip>

#define N 8000000
#define RANGE 100000
#define NUM_BINS 128

int main(int argc, char** argv) {

    // Random data set ranging from 1 to RANGE, inclusive
    int* data = new int[N];
    for (int i = 0; i < N; i++) {
        data[i] = rand() % RANGE + 1;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Count the number of unique nodes
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    int is_first_on_node = 1;
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);

    int node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    if (node_rank != 0) {
        is_first_on_node = 0;
    }

    int total_nodes = 0;
    MPI_Allreduce(&is_first_on_node, &total_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_free(&node_comm);

    // Calculate how wide each bin is (range of values it will cover)
    double bin_width = (double)RANGE / NUM_BINS;

    // Calculate how many elements each MPI process handles
    int elements_per_process = N / size;
    int start_idx = rank * elements_per_process;
    int end_idx = (rank == size - 1) ? N : start_idx + elements_per_process;

    // Start timing for histogram computation
    double start_time = MPI_Wtime();

    // Compute the local histogram for this process
    int local_histogram[NUM_BINS] = {0};
    for (int i = start_idx; i < end_idx; i++) {
        int bin = floor((data[i] - 1) / bin_width);
        if (bin >= NUM_BINS) {
            bin = NUM_BINS - 1; // If the value is greater than the last bin, put it in the last bin (if a flooring error occurs)
        }
        local_histogram[bin]++;
    }

    // Global histogram to collect results (only in rank 0, or the head node)
    int* global_histogram = NULL;
    if (rank == 0) {
        global_histogram = new int[NUM_BINS]();
    }

    // Reduce all local histograms to the global one
    MPI_Reduce(local_histogram, global_histogram, NUM_BINS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // End timing for histogram computation
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    // Gather timing stats
    double max_time, min_time, avg_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        avg_time /= size;
    }

    // Print the histogram in the head node
    if (rank == 0) {
        std::cout << "Parallel Histogram Results\n";
        std::cout << "============================\n\n";
        std::cout << "Total elements: " << N << " | Range: 1-" << RANGE << " | Number of Bins: " << NUM_BINS << "\n";
        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "Unique Nodes: " << total_nodes << "\n\n";
        std::cout << "Performance Metrics:\n";
        std::cout << "  └── Min Time: " << std::fixed << std::setprecision(10) << min_time << " seconds\n";
        std::cout << "  └── Max Time: " << std::fixed << std::setprecision(10) << max_time << " seconds\n";
        std::cout << "  └── Avg Time: " << std::fixed << std::setprecision(10) << avg_time << " seconds\n\n";

        for (int i = 0; i < NUM_BINS; i++) {
            int bin_start = i == 0 ? 1 : floor(i * bin_width) + 1;
            int bin_end = floor((i + 1) * bin_width);
            if (i == NUM_BINS - 1) bin_end = RANGE;
            
            std::cout << "Bin " << i << ": " << "[" << bin_start << " - " << bin_end << "]\n";
            std::cout << "  └── Count: " << global_histogram[i] << "\n";
        }
        delete[] global_histogram;
    }

    delete[] data;
    MPI_Finalize();
    return 0;
}