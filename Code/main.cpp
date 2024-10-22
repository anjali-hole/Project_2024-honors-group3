#include "data_generation.hpp"
#include "sorting_algorithms.hpp"
#include "mpi.h"
#include <cstdlib>  // For atoi
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);
    cali::ConfigManager mgr;
    mgr.start();

    // Read arguments
    int algToRun; // 0 is bitonic, 1 is sample, 2 is merge, 3 is radix, 4 is column
    int data_generation; // 0 is sorted, 1 is 1% perturbed, 2 is random, 3 is reverse sorted
    size_t array_size;
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
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* local_data = new int[array_size];
    
    // Add metadata collection
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    const char* algorithm_names[] = {"bitonic", "sample", "merge", "radix", "column"};
    const char* data_gen_names[] = {"Sorted", "1_perc_perturbed", "Random", "ReverseSorted"};
    
    adiak::value("algorithm", algorithm_names[algToRun]);
    adiak::value("programming_model", "mpi");
    adiak::value("data_type", "int");
    adiak::value("size_of_data_type", sizeof(int));
    adiak::value("input_size", array_size);
    adiak::value("input_type", data_gen_names[data_generation]);
    adiak::value("num_procs", comm_size);
    adiak::value("scalability", "strong"); // Adjust for weak scaling
    adiak::value("group_num", 3); 
    adiak::value("implementation_source", "handwritten");

    CALI_MARK_BEGIN("data_init_runtime");

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
    } else if (data_generation == 4) {
        generate_test_column(local_data, array_size, comm_size, rank);
    }
    else
    {
        printf("\n Please provide a valid data generation method");
        CALI_MARK_END("data_init_runtime");
        return 1;
    }
    CALI_MARK_END("data_init_runtime");


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
        CALI_MARK_END("main");
        return 1;
    }

    int* global_sorted_data = nullptr;
    if (rank == 0) {
        global_sorted_data = new int[array_size * comm_size];
    }    
    
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather(local_data, array_size, MPI_INT, global_sorted_data, array_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (rank == 0) {
        delete[] global_sorted_data; // Free allocated memory
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

    delete[] local_data;
    
    mgr.stop();
    mgr.flush();
    
    MPI_Finalize();
    
    return 0;
}
