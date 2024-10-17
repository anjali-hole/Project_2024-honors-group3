#include "sorting_algorithms.hpp"
#include "mpi.h"
#include <algorithm>
#include <bitset>
#include <caliper/cali.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

inline void sequential_sort(int* local_data, size_t local_data_size) {
    std::sort(local_data, local_data + local_data_size);
}

#pragma region bitonic_sort

// Basically the same as merge sort's merge, though this one keeps the output list in two arrays (to be send to different processes)
void bitonic_merge(int* data1, int* data2, int* smaller_half, int* larger_half, size_t length) {
    int index1 = 0;
    int index2 = 0;

    while (index1 < length && index2 < length) {
        int output_index = index1 + index2;
        bool choose_data1 = data1[index1] < data2[index2];

        int value;
        if (choose_data1) {
            value = data1[index1];
            index1++;
        } else {
            value = data2[index2];
            index2++;
        }

        if (output_index < length) {
            smaller_half[output_index] = value;
        } else {
            larger_half[output_index - length] = value;
        }
    }

    // by this point we are guaranteed to be filling larger_half, since we have completely gone through one of the input arrays
    while (index1 < length) {
        int output_index = index1 + index2;
        larger_half[output_index - length] = data1[index1];
        index1++;
    }

    while (index2 < length) {
        int output_index = index1 + index2;
        larger_half[output_index - length] = data2[index2];
        index2++;
    }
}

// Assumes the total list size is a power of 2, and that comm_size is a power or 2 less than or equal to the list size, and that local_data size times comm_size is the total list size
void bitonic_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    sequential_sort(local_data, local_data_size);

    for (int level = 0; level < std::log2(comm_size); level++) {
        std::bitset<32> rank_bitset(rank);
        bool is_increasing = !rank_bitset.test(level + 1);

        for (int current_bit = level; current_bit >= 0; current_bit--) {
            std::bitset<32> other_rank_bitset(rank_bitset);
            other_rank_bitset.flip(current_bit);
            int other_rank = other_rank_bitset.to_ulong();

            // While the data lives on two processes, only one needs to do the comparison.
            // For now the lower rank process will always do the comparison, though it might speed up the algorithm if we try to balance who does the comparison more evenly.
            bool is_doing_comparison = rank < other_rank;
            if (is_doing_comparison) {
                int* other_data = new int[local_data_size];
                int* smaller_half = new int[local_data_size];
                int* larger_half = new int[local_data_size];

                MPI_Recv(other_data, local_data_size, MPI_INT, other_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                bitonic_merge(local_data, other_data, smaller_half, larger_half, local_data_size);

                if (is_increasing) {
                    std::memcpy(local_data, smaller_half, local_data_size * sizeof(int));
                    MPI_Send(larger_half, local_data_size, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
                }
                else {
                    std::memcpy(local_data, larger_half, local_data_size * sizeof(int));
                    MPI_Send(smaller_half, local_data_size, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
                }

                delete[] other_data;
                delete[] smaller_half;
                delete[] larger_half;
            }
            else {
                MPI_Send(local_data, local_data_size, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
                MPI_Recv(local_data, local_data_size, MPI_INT, other_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

#pragma endregion

#pragma region sample_sort

void sample_sort(int*& local_data, size_t &local_data_size, int comm_size, int rank) {

    srand(time(NULL) * rank);
    constexpr size_t OVERSAMPLE_FACTOR = 3;
    
    //Take samples from the local data
    int local_samples[OVERSAMPLE_FACTOR];
    for (size_t i = 0; i < OVERSAMPLE_FACTOR; ++i)
        local_samples[i] = local_data[rand() % local_data_size];
    
    //Gather samples into process 0
    int* oversampled;
    int samples = OVERSAMPLE_FACTOR * comm_size;
    if (rank == 0) oversampled = new int[samples];
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Gather(local_samples, OVERSAMPLE_FACTOR, MPI_INT, oversampled, OVERSAMPLE_FACTOR, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");
    if (rank == 0)
    {
      std::cout << "Gathered samples: ";
      for (int i = 0; i < samples; ++i) std::cout << oversampled[i] << " ";
      std::cout << std::endl;
    }
    
    
    //Select splitters
    int* splitters = new int[comm_size];
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    if (rank == 0) {
        sequential_sort(oversampled, samples);
        for (size_t i = 1; i < comm_size; ++i)
            splitters[i-1] = oversampled[i * OVERSAMPLE_FACTOR];
        splitters[comm_size-1] = INT_MAX;
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");    
    MPI_Bcast(splitters, comm_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");
    if (rank == 0)
    {
      std::cout << "Gathered splitters: ";
      for (int i = 0; i < comm_size; ++i) std::cout << splitters[i] << " ";
      std::cout << std::endl;
    }
    
    //Split local data into buckets based on splitters
    std::vector<int>* buckets = new std::vector<int>[comm_size];
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (size_t i = 0; i < local_data_size; ++i) {
        int bucket = std::lower_bound(splitters, splitters + comm_size, local_data[i]) - splitters;
        buckets[bucket].push_back(local_data[i]);
    }
    
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    std::vector<int> local_bucket;
    for (int i = 0; i < comm_size; ++i) {
        if (i == rank) { //Receive data from others 
            for (int j = 0; j < comm_size; ++j) {
                if (i == j) local_bucket.insert(local_bucket.end(), buckets[i].begin(), buckets[i].end()); //Add this process' data
                else { //Receive from other processes
                    int element_count = 0;
                    MPI_Recv(&element_count, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (element_count != 0) {
                        int curr_size = local_bucket.size();
                        local_bucket.resize(curr_size + element_count);
                        MPI_Recv(&local_bucket[curr_size], element_count, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
        else { //Send to other processes
            int element_count = buckets[i].size();
            MPI_Send(&element_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            if (element_count != 0) MPI_Send(buckets[i].data(), element_count, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    sequential_sort(local_bucket.data(), local_bucket.size());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");  
    
    
    local_data_size = local_bucket.size();
    delete[] local_data;
    local_data = new int[local_data_size];
    memcpy(local_data, local_bucket.data(), local_data_size * sizeof (int));
    
    //testing
    //std::cout << "Rank " << rank << " after local sort: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //        std::cout << local_data[i] << " ";
    // }
    //  std::cout << std::endl;
    
    delete[] buckets;
    delete[] splitters;
    if (rank == 0) delete[] oversampled;
    
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

    //testing
    /*std::cout << "Rank " << rank << " initial data: ";
    for (size_t i = 0; i < local_data_size; ++i) {
        std::cout << local_data[i] << " ";
    }
    std::cout << std::endl; 
    */

    //sort local data
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    sequential_sort(local_data, local_data_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");


    //testing
    /*std::cout << "Rank " << rank << " after local sort: ";
     for (size_t i = 0; i < local_data_size; ++i) {
            std::cout << local_data[i] << " ";
      }
      std::cout << std::endl;
    */
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
	    /*std::cout << "Rank " << rank << " received from " << partner << ": ";
            for (size_t i = 0; i < partner_size; ++i) {
                std::cout << received_data[i] << " ";
            }
            std::cout << std::endl;
 	    */

           //merge
            std::vector<int> merged(local_data_size + partner_size);   
            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_small");
            merge(local_data, local_data_size, received_data.data(), partner_size, merged.data());
            CALI_MARK_END("comp_small");
            CALI_MARK_END("comp");


            //keep correct half of the data
            size_t new_size = (rank < partner) ? std::min(local_data_size, merged.size() / 2) : std::max(local_data_size, merged.size() / 2);
            std::copy(merged.begin() + (rank < partner ? 0 : merged.size() - new_size), 
                      merged.begin() + (rank < partner ? new_size : merged.size()), 
                      local_data);
            local_data_size = new_size;            


  	    //testing
	    /*std::cout << "Rank " << rank << " after merge with " << partner << ": ";
            for (size_t i = 0; i < local_data_size; ++i) {
                std::cout << local_data[i] << " ";
            }
            std::cout << std::endl;
	    */
        }
    }
   
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

void radix_sort(int*& local_data, size_t &local_size, int comm_size, int rank) {
    // Transfer data between processes such that each process has a correct range of values
    int global_max, global_min;
    int local_max = *std::max_element(local_data, local_data + local_size);
    int local_min = *std::min_element(local_data, local_data + local_size);
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    int* send_counts = new int[comm_size]();
    int* send_offsets = new int[comm_size]();
    int* recv_counts = new int[comm_size]();
    int* recv_offsets = new int[comm_size]();

    // calculate the range of values that each process will receive and send
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
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int i = 0; i < comm_size; i++) {
        recv_offsets[i] = total_recv;
        total_recv += recv_counts[i];
    }

    int* recv_data = new int[total_recv];
    MPI_Alltoallv(send_data, send_counts, send_offsets, MPI_INT, recv_data, recv_counts, recv_offsets, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    delete[] local_data;
    local_data = new int[total_recv];
    local_size = total_recv;
    std::copy(recv_data, recv_data + total_recv, local_data);

    // radix sort now that all data are in correct processes
    CALI_MARK_BEGIN("comp");
    int max_val = *std::max_element(local_data, local_data + total_recv);
    for (long int exp = 1; max_val / exp > 0; exp *= 10)
    {
        CALI_MARK_BEGIN("comp_large");
        counting_sort(local_data, total_recv, exp);
        CALI_MARK_END("comp_large");
    }
    CALI_MARK_END("comp");

    // print data
    // for (int i = 0; i < comm_size; i++) {
    //     if (rank == i) {
    //         std::cout << "Rank " << rank << " size: " << total_recv << " data: ";
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
}

#pragma endregion

#pragma region column_sort

void column_sort(int* local_data, size_t local_data_size, int comm_size, int rank) {
    std::cout << "LOCLA DATS SIZE IS " << local_data_size << " and COMM SIZE IS " << comm_size << std::endl;
    // local_data is the column
    CALI_MARK_BEGIN("whole_column_sort");
    int s = comm_size;
    int r = local_data_size;

    //testing
    // std::cout << "(Pre Step 1)Rank " << rank << " initial data: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl; 
    // step 1: sort column
    sequential_sort(local_data, local_data_size);
    //testing
    // std::cout << "(Post Step 1)Rank " << rank << " initial data: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl; 

    // step 2: Transpose: access values in CMO and place them back into matrix in RMO
    int* send_buf = new int[local_data_size];
    int subbuf_size = local_data_size / comm_size;
    for(int i = 0; i < local_data_size; ++i) {
        int process = i % comm_size;
        // std::cout << "Rank " << rank << " inserting " << i << " into " << (subbuf_size * process) + (i / comm_size) << std::endl;
        send_buf[(subbuf_size * process) + (i / comm_size)] = local_data[i];
    }
    // std::cout << "send_buf for rank " << rank << ":"; 
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << send_buf[i] << " ";
    // }
    // std::cout << std::endl;
    MPI_Alltoall(send_buf, subbuf_size, MPI_INT, local_data, subbuf_size, MPI_INT, MPI_COMM_WORLD);
    delete[] send_buf;

    //testing
    // std::cout << "(Post Step 2)Rank " << rank << " initial data: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl; 

    // step 3: sort "column"
    sequential_sort(local_data, local_data_size);

    // step 4: "untranspose"
    MPI_Alltoall(MPI_IN_PLACE, subbuf_size, MPI_INT, local_data, subbuf_size, MPI_INT, MPI_COMM_WORLD);

    //testing
    // std::cout << "(Post Step 4)Rank " << rank << " initial data: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl; 

    // step 5: sort column
    sequential_sort(local_data, local_data_size);
    //testing
    // std::cout << "(Post step 5)Rank " << rank << " initial data: ";
    // for (size_t i = 0; i < local_data_size; ++i) {
    //     std::cout << local_data[i] << " ";
    // }
    // std::cout << std::endl; 

    // step 6: "shift"
    // we shift whats needed to the correct process, but not in the order recommended
    // 0->1, 1->2, 2->0
        int shift_buf_size = (local_data_size / 2) * comm_size;
        int* shift_buf = new int[shift_buf_size]();

        // Fill shift buf such that values that are sent to target proc are filled appropriately in their section, rest are 0
        int half_local_size_ceil = (local_data_size + 1) / 2;
        int start = half_local_size_ceil;
        int target_rank = rank + 1;
        if (rank == comm_size - 1) {
            target_rank = 0;
        }
        int offset = (local_data_size / 2) * target_rank;

        for(int i = start; i < local_data_size; ++i) {
            shift_buf[offset + (i - half_local_size_ceil)] = local_data[i];
        }

        // Receive array of (local_data_size / 2) * comm_size, get appropriate elements by offset place them at top of local_data
        int* receive_buf = new int[shift_buf_size]();
        MPI_Alltoall(shift_buf, shift_buf_size / comm_size, MPI_INT, receive_buf, shift_buf_size / comm_size, MPI_INT, MPI_COMM_WORLD);
        int receive_rank = rank - 1;
        if (rank == 0) {
            receive_rank = comm_size - 1;
        }
        std::cout << std::endl;
        offset = receive_rank * (local_data_size / 2);
        for(int i = half_local_size_ceil; i < local_data_size; ++i) {
            local_data[i] = receive_buf[offset + i - half_local_size_ceil];
        }

        // testing
        // std::cout << "(Post step 6) Rank " << rank << " data: ";
        // for (size_t i = 0; i < local_data_size; ++i) {
        //     std::cout << local_data[i] << " ";
        // }
        // std::cout << std::endl;

    // step 7: everyone except process 0 sequential sort
    if (rank != 0){
        sequential_sort(local_data, local_data_size);
    }

    // step 8: "unshift"
    // shift back: 2->1, 1->0, 0->2
        delete[] shift_buf;
        delete[] receive_buf;
        shift_buf = new int[shift_buf_size]();
        receive_buf = new int[shift_buf_size]();

        if (rank != 0) {
            // for non-first columns, select first half of elements to move to prev column
            target_rank = rank - 1;
            offset = (local_data_size / 2) * target_rank;
            for(int i = 0; i < local_data_size / 2; ++i) {
                shift_buf[offset + i] = local_data[i];
            }
        } else {
            // for last column, select second half of elements to move to last column
            target_rank = comm_size - 1;
            offset = (local_data_size / 2) * target_rank;
            for(int i = half_local_size_ceil; i < local_data_size; ++i) {
                shift_buf[offset + (i - half_local_size_ceil)] = local_data[i];
            }
        }

        MPI_Alltoall(shift_buf, shift_buf_size / comm_size, MPI_INT, receive_buf, shift_buf_size / comm_size, MPI_INT, MPI_COMM_WORLD);

        if (rank != 0) { // no shifting needs to be done for first column
            int* temp = new int[local_data_size]();
            for(int i = half_local_size_ceil; i < local_data_size; ++i) {
                temp[i - half_local_size_ceil] = local_data[i];
            }
            delete[] local_data;
            local_data = temp;
        }
        receive_rank = rank + 1; // receiving from next column
        if (rank == comm_size - 1) {
            receive_rank = 0; // except for last column, that receives from column0
        }
        offset = receive_rank * shift_buf_size / comm_size; // offset in receive buf
        for(int i = 0; i < local_data_size; ++i) {
            local_data[half_local_size_ceil + i] = receive_buf[offset + i];
        }

        // testing
        if (rank == 0) {
            std::cout << "(Post step 8) Rank " << rank << " data: ";
            for (size_t i = 0; i < local_data_size; ++i) {
                std::cout << local_data[i] << "-";
            }
            std::cout << std::endl;
        } else if (rank == 1) {
            std::cout << "(Post step 8) Rank " << rank << " data: ";
            for (size_t i = 0; i < local_data_size; ++i) {
                std::cout << local_data[i] << "+";
            }
            std::cout << std::endl;
        } else {
            std::cout << "(Post step 8) Rank " << rank << " data: ";
            for (size_t i = 0; i < local_data_size; ++i) {
                std::cout << local_data[i] << "_";
            }
            std::cout << std::endl;
        }

    delete[] shift_buf;
    delete[] receive_buf;
    
    CALI_MARK_END("whole_column_sort");
}

#pragma endregion
