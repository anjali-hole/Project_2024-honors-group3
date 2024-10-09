void generate_sorted(int* local_data, size_t local_data_size, int comm_size, int rank);

void generate_sorted_1percent_perturbed(int* local_data, size_t local_data_size, int comm_size, int rank);

void generate_random(int* local_data, size_t local_data_size, int comm_size, int rank);

void generate_reverse_sorted(int* local_data, size_t local_data_size, int comm_size, int rank);

bool check_data_sorted(int* local_data, size_t local_data_size, int comm_size, int rank);