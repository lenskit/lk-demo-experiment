#!/bin/bash
#SBATCH -J lkpy
#SRUN -J lkpy

node=$(hostname)
echo "Running job on node $node" >&2

# Boise State's SLURM cluster has aggresive ulimits, even with larger job requests
# Reset those limits
ulimit -v unlimited
ulimit -u 2048
ulimit -n 4096

# Configure LensKit threading based on SLURM
cpus="$SLURM_CPUS_ON_NODE"

if [ -z "$LK_NUM_PROCS" -a -n "$cpus" ]; then
    # set process count from SLURM
    procs=$(expr $cpus / 2)
    if [[ $procs = 0 ]]; then
        $procs=1
    fi
    echo "using $procs LK processes"
    export LK_NUM_PROCS=$procs
fi

if [ -n "$cpus" ]; then
    echo "using $cpus Numba threads"
    export NUMBA_NUM_THREADS="$cpus"
    export MKL_NUM_THREADS=1
fi

# this really works best
export MKL_THREADING_LAYER=tbb

# Finally run the code
exec "$@"
