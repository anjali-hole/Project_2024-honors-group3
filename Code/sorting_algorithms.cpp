#include "sorting_algorithms.hpp"
#include <iostream>
#include "mpi.h"
#include <caliper/cali.h>
#include <algorithm>
#include <vector>
#include <climits>

void sequential_sort(int* local_data, size_t local_data_size) {
    CALI_MARK_BEGIN("comp_small");
    std::sort(local_data, local_data + local_data_size);
    CALI_MARK_END("comp_small");
}

#pragma region bitonic_sort

void bitonic_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {

}

#pragma endregion

#pragma region sample_sort

/*
int** partition(int* data, size_t data_size, double oversample, int bucket_count)
{
    srand(time(NULL));
    size_t sample_count = oversample * bucket_count;
    int* samples = new int[sample_count];
    for (size_t i = 0; i < sample_count; ++i)
    {
        samples[i] = data[rand() % data_size];
    }
    
}
*/ 

void sample_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    srand(time(NULL) * rank);
    constexpr size_t OVERSAMPLE_FACTOR = 3;
    
    int local_samples[OVERSAMPLE_FACTOR];
    for (size_t i = 0; i < OVERSAMPLE_FACTOR; ++i)
        local_samples[i] = local_data[rand() % local_data_size];

    int* oversampled;
    int* splitters = new int[comm_size];
    int samples = OVERSAMPLE_FACTOR * comm_size;
    if (rank == 0) oversampled = new int[samples];
    MPI_Gather(local_samples, OVERSAMPLE_FACTOR, MPI_INT, oversampled, samples, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        sequential_sort(oversampled, samples);
        for (size_t i = 1; i < comm_size; ++i)
            splitters[i-1] = oversampled[i * OVERSAMPLE_FACTOR];
    }
    splitters[comm_size-1] = INT_MAX;
    MPI_Bcast(splitters, comm_size, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int>* buckets = new std::vector<int>[comm_size]; 
    
    for (size_t i = 0; i < local_data_size; ++i) {
        int bucket = std::lower_bound(splitters, splitters + comm_size, local_data[i]) - splitters;
        buckets[bucket].push_back(i);
    }

    std::vector<int> local_bucket;
    int copy_val;
    for (int i = 0; i < comm_size; ++i) {
        if (i == rank) { //Receive data from others 
            for (int j = 0; i < comm_size; ++i) {
                if (i == j) for (auto local_val : buckets[i]) local_bucket.push_back(local_val);
                else {
                    //Receive element count
                    //Receive elements
                } 
            }
        }
        else {
            //Send element count
            //Send elements
        }
    }
    
    sequential_sort(local_bucket.data(), local_bucket.size());
}

#pragma endregion

#pragma region merge_sort

// Helper function for merging two sorted arrays
void merge(int* left, int left_size, int* right, int right_size, int* result) {
    CALI_MARK_BEGIN("comp_small");
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
    CALI_MARK_END("comp_small");
}

void merge_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("comp_large");

    //testing
    std::cout << "Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 


   //sort local data
    sequential_sort(local_data, local_data_size);


    //testing
    std::cout << "Rank " << rank << " after local sort: ";
     for (size_t i = 0; i < local_data_size; ++i) {
            std::cout << local_data[i] << " ";
      }
      std::cout << std::endl;

    //parallel mergee
    for (int step = 1; step < comm_size; step *=2){
        int partner = rank ^ step;

        if (partner < comm_size){
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_small");
            int partner_size;
            MPI_Sendrecv(&local_data_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END("comm_small");
            CALI_MARK_END("comm");

            //buffer for recieved data
            std::vector<int> received_data(partner_size);
            
            //exchange data
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            MPI_Sendrecv(local_data, local_data_size, MPI_INT, partner, 1, received_data.data(), partner_size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

//testing
	    std::cout << "Rank " << rank << " received from " << partner << ": ";
            for (size_t i = 0; i < partner_size; ++i) {
                std::cout << received_data[i] << " ";
            }
            std::cout << std::endl;
 

           //merge
            std::vector<int> merged(local_data_size + partner_size);   
            merge(local_data, local_data_size, received_data.data(), partner_size, merged.data());
               


            //keep correct half of the data
            size_t new_size = (rank < partner) ? std::min(local_data_size, merged.size() / 2) : std::max(local_data_size, merged.size() / 2);
            std::copy(merged.begin() + (rank < partner ? 0 : merged.size() - new_size), 
                      merged.begin() + (rank < partner ? new_size : merged.size()), 
                      local_data);
            local_data_size = new_size;            


	//testing
	    std::cout << "Rank " << rank << " after merge with " << partner << ": ";
            for (size_t i = 0; i < local_data_size; ++i) {
                std::cout << local_data[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    //testing
    std::cout << "Rank " << rank << " final data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl;
 
   CALI_MARK_END("comp_large");
}

#pragma endregion

#pragma region radix_sort

void counting_sort(int* arr, int n, int exp) {
    int* output = new int[n];
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    delete[] output;
}

void radix_sort(int* local_data, size_t local_size, int comm_size, int rank) {
    CALI_MARK_BEGIN("whole_radix_sort");
    // Find the maximum value in the local data to determine the number of digits and count sort each digit
    int max_val = *std::max_element(local_data, local_data + local_size);
    for (int exp = 1; max_val / exp > 0; exp *= 10)
    {
        CALI_MARK_BEGIN("counting_sort");
        counting_sort(local_data, local_size, exp);
        CALI_MARK_END("counting_sort");
    }
    
    // Transfer data between processes such that data is sorted across all processes
    CALI_MARK_BEGIN("comm_full");
    int global_max, global_min;
    int local_max = *std::max_element(local_data, local_data + local_size);
    int local_min = *std::min_element(local_data, local_data + local_size);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    CALI_MARK_BEGIN("comm_send");
    int* send_counts = new int[comm_size]();
    int* send_offsets = new int[comm_size]();
    int* recv_counts = new int[comm_size]();
    int* recv_offsets = new int[comm_size]();

    int range_size = (global_max - global_min + 1) / comm_size;

    std::vector<std::vector<int>> buckets(comm_size);
    for (int i = 0; i < local_size; i++) {
        int value = local_data[i];
        int target_proc = (value - global_min) / range_size;
        if (target_proc >= comm_size) {
            target_proc = comm_size - 1;
        }
        buckets[target_proc].push_back(value);
    }

    int total_send = 0;
    for (int i = 0; i < comm_size; i++) {
        send_counts[i] = buckets[i].size();
        send_offsets[i] = total_send;
        total_send += send_counts[i];
    }

    int* send_data = new int[total_send];
    int index = 0;
    for (int i = 0; i < comm_size; i++) {
        for (size_t j = 0; j < buckets[i].size(); j++) {
            send_data[index++] = buckets[i][j];
        }
    }

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("comm_send");

    CALI_MARK_BEGIN("comm_recv");
    int total_recv = 0;
    for (int i = 0; i < comm_size; i++) {
        recv_offsets[i] = total_recv;
        total_recv += recv_counts[i];
    }

    int* recv_data = new int[total_recv];
    MPI_Alltoallv(send_data, send_counts, send_offsets, MPI_INT, recv_data, recv_counts, recv_offsets, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("comm_recv");

    std::copy(recv_data, recv_data + total_recv, local_data);
    CALI_MARK_END("comm_full");

    max_val = *std::max_element(local_data, local_data + total_recv);
    for (int exp = 1; max_val / exp > 0; exp *= 10)
        counting_sort(local_data, total_recv, exp);

    // print data
    // for (int i = 0; i < comm_size; i++) {
    //     if (rank == i) {
    //         std::cout << "Rank " << rank << " data: ";
    //         for (int j = 0; j < total_recv; j++) {
    //             std::cout << local_data[j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    delete[] send_counts;
    delete[] send_offsets;
    delete[] recv_counts;
    delete[] recv_offsets;
    delete[] send_data;
    delete[] recv_data;
    CALI_MARK_END("whole_radix_sort");
}

#pragma endregion

#pragma region column_sort

void column_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    // local_data is the column
    CALI_MARK_BEGIN("whole_column_sort");
    int s = comm_size;
    int r = local_data_size;

    //testing
    std::cout << "(Pre Step 1)Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 

    // step 1: sort column
    sequential_sort(local_data, local_data_size);
    //testing
    std::cout << "(Post Step 1)Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 

    // step 2: Transpose: access values in CMO and place them back into matrix in RMO
    int* transposed_data = new int[local_data_size];
    MPI_Alltoall(local_data, local_data_size, MPI_INT, transposed_data, local_data_size, MPI_INT, MPI_COMM_WORLD);
    std::copy(transposed_data, transposed_data + local_data_size, local_data);
    delete[] transposed_data;

    //testing
    std::cout << "(Post Step 2)Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 

    // step 3: sort "column"
    sequential_sort(local_data, local_data_size);

    // step 4: "untranspose"
    std::vector<MPI_Request> send_requests(comm_size);
    std::vector<MPI_Request> recv_requests(comm_size);
    int* untransposed_data = new int[local_data_size];
    for (int i = 0; i < comm_size; i++) {
        int send_index = i * local_data_size;
        int recv_index = i * local_data_size;
        MPI_Isend(local_data + send_index, local_data_size, MPI_INT, i, 0, MPI_COMM_WORLD, &send_requests[i]);
        MPI_Irecv(untransposed_data + recv_index, local_data_size, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
    }
    MPI_Waitall(s, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(s, recv_requests.data(), MPI_STATUSES_IGNORE);
    std::copy(untransposed_data, untransposed_data + local_data_size, local_data);
    delete[] untransposed_data;

    //testing
    std::cout << "(Post Step 4)Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 

    // step 5: sort column
    sequential_sort(local_data, local_data_size);
    //testing
    std::cout << "(Post step 5)Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 
    
    CALI_MARK_END("whole_column_sort");
}

#pragma endregion
