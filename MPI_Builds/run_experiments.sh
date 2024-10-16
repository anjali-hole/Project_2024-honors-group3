#!/bin/bash

#Array for input sizes
# TODO: Uncomment the line below for actual sizes
# input_sizes=(65536 262144 1048576 4194304 16777216 67108864 268435456)
# input_sizes=(8 16 32 64 128 256 512)
input_sizes=(32)

# Array of input types
# 0: Sorted, 1: 1% perturbed, 2: Random, 3: Reverse sorted
# TODO: Add 1 and 2 after those functions are implemented
# input_types=(0 3)
input_types=(2)

# Array of number of processors
# num_procs=(2 4 8 16 32 64 128 256 512 1024)
num_procs=(4)

# Merge sort algorithm index
#TODO: change this based on algorithm
alg=4

# Maximum processes per node
max_procs_per_node=48

# Create directories for output
mkdir -p slurm_output
mkdir -p caliper_output

# Function to submit a job
submit_job() {
    local size=$1
    local type=$2
    local procs=$3

    # Calculate number of nodes needed
    local nodes=$(( ($procs + $max_procs_per_node - 1) / $max_procs_per_node ))
    local tasks_per_node=$(( $procs < $max_procs_per_node ? $procs : $max_procs_per_node ))

    # Create a temporary job script
    cat << EOF > temp_job.sh
#!/bin/bash
#SBATCH --export=NONE
#SBATCH --get-user-env=L
#SBATCH --job-name=MergeSort_${size}_${type}_${procs}
#SBATCH --time=03:00:00
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --mem=32G
#SBATCH --output=slurm_output/output_mergesort_${size}_${type}_${procs}.%j

module load intel/2020b
module load CMake/3.12.1
module load GCCcore/8.3.0

CALI_CONFIG="spot(output=caliper_output/result_mergesort_${size}_${type}_${procs}.cali, time.variance,profile.mpi)"
mpirun -np $procs ./main $alg $type $size
EOF

    # Submit the job
    sbatch temp_job.sh

    # Remove the temporary job script
    rm temp_job.sh
}

# Main loop to submit all jobs
for size in "${input_sizes[@]}"; do
    for type in "${input_types[@]}"; do
        for procs in "${num_procs[@]}"; do
            echo "Submitting merge sort job: size=$size, type=$type, procs=$procs"
            submit_job $size $type $procs

            # Small delay to avoid overwhelming the job scheduler
            sleep 1
        done
    done
done

echo "All merge sort jobs submitted."
