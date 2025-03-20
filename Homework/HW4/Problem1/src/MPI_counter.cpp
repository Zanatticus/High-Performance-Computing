// A program that increments/decrements a counter after sending it to another MPI process
// Author: Zander Ingare

#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    int world_rank, world_size;
    const int COUNT_LIMIT = 64;
    int counter = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Get_processor_name(processor_name, &processor_name_len);


    
    // Print total number of ranks from process 0
    if (world_rank == 0) {
        std::cout << "Total MPI ranks allocated: " << world_size << std::endl;
    }
    
    // Process 0 starts the counter
    if (world_rank == 0) {
        counter = 0;  // Start with 1
        std::cout << "Process " << world_rank << " has counter value " << counter << std::endl;
        counter++;  // Increment before sending
        MPI_Send(&counter, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
    }

    while (counter < COUNT_LIMIT) {
        // All other processes receive from previous rank
        if (world_rank != 0) {
            MPI_Recv(&counter, 1, MPI_INT, (world_rank - 1 + world_size) % world_size, 0, MPI_COMM_WORLD, &status);
            std::cout << "Process " << world_rank << " has counter value " << counter << std::endl;
            
            // Only increment and send if we haven't reached the limit
            if (counter < COUNT_LIMIT) {
                counter++;
                // Send to next process (wrap around to 0 if we're the last process)
                MPI_Send(&counter, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}