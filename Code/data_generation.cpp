#include "data_generation.hpp"
#include <iostream>
#include "mpi.h"
#include <caliper/cali.h>
#include <climits>
void generate_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("data_init_runtime");
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

    CALI_MARK_END("data_init_runtime");
}

void generate_test_column(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("data_init_runtime");
    // Calculate the global start index for this rank
    // int matrix[27] = {
    //     14, 3, 21, 24, 8, 26, 19, 10, 2,   // First column
    //     25, 1, 12, 23, 13, 4, 17, 15, 20,  // Second column
    //     9, 27, 7, 16, 18, 5, 22, 11, 6     // Third column
    // };

    int matrix[32] = {
        14, 27, 8, 16, 1, 12, 25, 6, 9,   // First column
        10, 30, 21, 5, 24, 11, 4, 17, 18, // Second column
        22, 32, 19, 7, 2, 31, 15, 26, 13, // Third column
        23, 28, 20, 3, 29                 // Fourth column
    };

    size_t global_start = rank * local_data_size;
    
    // Generate sorted data for this rank
    for (size_t i = 0; i < local_data_size; ++i) {
        local_data[i] = matrix[global_start + i];
    }
    //testing
    //std::cout << std::endl;
    //std::cout << "Rank " << rank << " generated data: ";
    //for (size_t i = 0; i < local_data_size; ++i) {
    //    std::cout << local_data[i] << " ";
    //}
    //std::cout << std::endl;

    CALI_MARK_END("data_init_runtime");
}

void generate_sorted_1percent_perturbed(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("data_perturbed_init_runtime");
    
    // Generate sorted data
    generate_sorted(local_data, local_data_size, comm_size, rank);
    srand(time(NULL) + rank);

    // Get 1% of local_data_size elements to swap
    size_t swap_count = local_data_size / 100;

    for (size_t i = 0; i < swap_count; ++i) {
        // Select two random indices to swap
        size_t index1 = rand() % local_data_size;
        size_t index2 = rand() % local_data_size;
        while (index2 == index1) {
            index2 = rand() % local_data_size;
        }
        // Swap
        int temp = local_data[index1];
        local_data[index1] = local_data[index2];
        local_data[index2] = temp;
    }
    CALI_MARK_END("data_perturbed_init_runtime");
}

void generate_random(int* local_data, size_t local_data_size, int comm_size, int rank) { 
    CALI_MARK_BEGIN("data_init_runtime");
    srand(time(NULL) * rank);
    for (size_t i = 0; i < local_data_size; ++i) {
        local_data[i] = rand();
    }

    //testing
    /*std::cout << std::endl;
    std::cout << "Rank " << rank << " generated data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }           
    std::cout << std::endl;*/
    CALI_MARK_END("data_init_runtime"); 
    
}

void generate_reverse_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("data_init_runtime");
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
    CALI_MARK_END("data_init_runtime");
}

bool check_data_sorted(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("correctness_check");
    // Check that all the local data is sorted
    for (size_t i = 0; i < local_data_size - 1; i++) {
        if (local_data[i] > local_data[i + 1]) {
            std::cout << "Local data failed " << local_data[i] << " > " << local_data[i+1] << std::endl;
            CALI_MARK_END("correctness_check");
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
            std::cout << "Gap failed " << local_data[local_data_size - 1] << " > " << receivedNum <<std::endl;
            CALI_MARK_END("correctness_check");
            return false;
        }
    }
    CALI_MARK_END("correctness_check");
    return true;
}

