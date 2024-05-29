#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=GPUComp_H2
#SBATCH --output=./outputs/GPUComp_H2-%j.out
#SBATCH --error=./errors/GPUComp_H2-%j.err
module load cuda

srun ./bin/GPUComp_H2 4 2
