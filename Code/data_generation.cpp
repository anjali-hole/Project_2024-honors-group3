#include "data_generation.hpp"
#include <iostream>
#include "mpi.h"

void generate_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    // Calculate the global start index for this rank
    size_t global_start = rank * local_data_size;
    
    // Generate sorted data for this rank
    for (size_t i = 0; i < local_data_size; ++i) {
        local_data[i] = global_start + i;
    }
    //testing
    //std::cout << std::endl;
    //std::cout << "Rank " << rank << " generated data: ";
    //for (size_t i = 0; i < local_data_size; ++i) {
    //    std::cout << local_data[i] << " ";
    //}
    //std::cout << std::endl;
}

void generate_sorted_1percent_perturbed(int* local_data, size_t local_data_size, int comm_size, int rank) {

}

void generate_random(int* local_data, size_t local_data_size, int comm_size, int rank) {

}

void generate_reverse_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    // Calculate the global end index for this rank
    size_t global_end = (comm_size - rank) * local_data_size;
    
    // Generate reverse sorted data for this rank
    for (size_t i = 0; i < local_data_size; ++i) {
        local_data[i] = global_end - i - 1;
    }
    //testing
    //std::cout << std::endl;
    //std::cout << "Rank " << rank << " generated data: ";
    //for (size_t i = 0; i < local_data_size; ++i) {
    //    std::cout << local_data[i] << " ";
    //}           
    //std::cout << std::endl;
}

bool check_data_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    // Check that all the local data is sorted
    for (size_t i = 0; i < local_data_size - 1; i++) {
        if (local_data[i] > local_data[i + 1]) {
            return false;
        }
    }

    // If not the first process send the smallest number down
    if (rank != 0) {
	MPI_Send(&local_data[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);      
    }

    // If not the last process make sure the number received is larger than the largest number in the local data
    if (rank != comm_size - 1) {
        int receivedNum;
        MPI_Recv(&receivedNum, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (local_data[local_data_size - 1] > receivedNum) {
            return false;
        }
    }

    return true;
}
