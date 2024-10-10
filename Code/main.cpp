#include "data_generation.hpp"
#include "sorting_algorithms.hpp"
#include "mpi.h"

int main(int argc, char *argv[]) {
    // Read arguments
    int algToRun; // 0 is bitonic, 1 is sample, 2 is merge, 3 is radix, 4 is column
    int data_generation; // 0 is sorted, 1 is 1% perturbed, 2 is random, 3 is reverse sorted
    int array_size;
    if (argc == 4)
    {
        algToRun = atoi(argv[1]);
        data_generation = atoi(argv[2]);
        array_size = atoi(argv[3]);
    }
    else
    {
        printf("\n Please an algorithm to run, data generation method, and array size");
        return 1;
    }

    // Get comm_size, rank, and allocate array for local data
    int comm_size, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* local_data = new int[array_size];

    if (data_generation == 0)
    {
        generate_sorted(local_data, array_size, comm_size, rank);
    }
    else if (data_generation == 1)
    {
        generate_sorted_1percent_perturbed(local_data, array_size, comm_size, rank);
    }
    else if (data_generation == 2)
    {
        generate_random(local_data, array_size, comm_size, rank);
    }
    else if (data_generation == 3)
    {
        generate_reverse_sorted(local_data, array_size, comm_size, rank);
    }
    else
    {
        printf("\n Please provide a valid data generation method");
        return 1;
    }

    // Call specified sorting algorithm
    if (algToRun == 0)
    {
        bitonic_sort(local_data, array_size, comm_size, rank);
    }
    else if (algToRun == 1)
    {
        sample_sort(local_data, array_size, comm_size, rank);
    }
    else if (algToRun == 2)
    {
        merge_sort(local_data, array_size, comm_size, rank);
    }
    else if (algToRun == 3)
    {
        radix_sort(local_data, array_size, comm_size, rank);
    }
    else if (algToRun == 4)
    {
        column_sort(local_data, array_size, comm_size, rank);
    }
    else
    {
        printf("\n Please provide a valid sorting algorithm");
        return 1;
    }

    // Check that data is sorted
    if (check_data_sorted(local_data, array_size, comm_size, rank))
    {
        printf("\n Data is sorted");
    }
    else
    {
        printf("\n Data is not sorted");
    }

    MPI_Finalize();
    return 0;
}