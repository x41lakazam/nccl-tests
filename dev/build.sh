#!/bin/sh
module purge

source dev/load_modules.sh

echo "NCCL_HOME=$NCCL_HOME"
echo "CUDA_HOME=$CUDA_HOME"

######
export DEBUG=1
make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME DEBUG=1