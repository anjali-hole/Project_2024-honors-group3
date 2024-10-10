#include "sorting_algorithms.hpp"

#include "mpi.h"

#include <algorithm>
#include <vector>

void sequential_sort(int* local_data, size_t local_data_size) {
    std::sort(local_data, local_data + local_data_size);
}

#pragma region bitonic_sort

void bitonic_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {

}

#pragma endregion

#pragma region sample_sort

void sample_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    
}

#pragma endregion

#pragma region merge_sort

// Helper function for merging two sorted arrays
void merge(int* left, int left_size, int* right, int right_size, int* result) {
    int i = 0, j = 0, k = 0;
    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }
    while (i < left_size) {
        result[k++] = left[i++];
    }
    while (j < right_size) {
        result[k++] = right[j++];
    }
}

void merge_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    //sort local data
    sequential_sort(local_data, local_data_size);

    //parallel mergee
    for (int step = 1; step < comm_size; step *=2){
        int partner = rank ^ step;

        if (partner < comm_size){
            int partner_size;
            MPI_Sendrecv(&local_data_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            //buffer for recieved data
            std::vector<int> received_data(partner_size);
            
            //exchange data
            MPI_Sendrecv(local_data, local_data_size, MPI_INT, partner, 1, received_data.data(), partner_size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //merge
            std::vector<int> merged(local_data_size + partner_size);
            if (rank < partner) {
                merge(local_data, local_data_size, received_data.data(), partner_size, merged.data());
            } else {
                merge(received_data.data(), partner_size, local_data, local_data_size, merged.data());
            }

            //keep correct half of the data
            size_t new_size = (rank < partner) ? local_data_size : partner_size;
            std::copy(merged.begin(), merged.begin() + new_size, local_data);
            local_data_size = new_size;
        }
    }
}

#pragma endregion

#pragma region radix_sort

void radix_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    
}

#pragma endregion

#pragma region column_sort

void column_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    
}

#pragma endregion
