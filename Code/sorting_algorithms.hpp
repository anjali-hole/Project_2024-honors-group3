#include <stddef.h>

void sequential_sort(int* local_data, size_t local_data_size);

void bitonic_sort(int* local_data, size_t local_data_size, int comm_size, int rank);

void sample_sort(int* local_data, size_t local_data_size, int comm_size, int rank);

void merge_sort(int* local_data, size_t local_data_size, int comm_size, int rank);

void radix_sort(int* local_data, size_t local_data_size, int comm_size, int rank);

void column_sort(int* local_data, size_t local_data_size, int comm_size, int rank);
