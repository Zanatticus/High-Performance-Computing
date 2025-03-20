// A program that increments/decrements a counter after sending it to another MPI process
// Author: Zander Ingare

#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    const int COUNT_LIMIT = 64;
    int counter = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &processor_name_len);

    if (size < COUNT_LIMIT) {
        if (rank == 0) {
            std::cout << "Error: Expected " << COUNT_LIMIT << " MPI ranks, but got " << size << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        counter = 1;
        std::cout << "Process " << rank << " on node " << processor_name << " started with counter: " << counter << "\n";
        counter++;
        MPI_Send(&counter, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&counter, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        std::cout << "Process " << rank << " on node " << processor_name << " received counter: " << counter << "\n";

        if (rank < COUNT_LIMIT - 1) {
            counter++;
            MPI_Send(&counter, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}