#!/bin/bash
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00
#PBS -q gpu
#PBS -A EMEWS

module load cmake/3.30.2-ufv3dko nvhpc/25.9/nvhpc cuda/13.0.0

export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_PATH/include:$CPATH

cd /home/erkoning/FLAMEGPU2-circles-benchmark/build_tmp
./bin/Release/circles-benchmark
