#!/bin/bash
#SBATCH --job-name=bert-dist
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -C A4000
#SBATCH --output=bert_%j.out
#SBATCH --error=bert_%j.err

export GLOO_SOCKET_IFNAME=eth4
export GLOO_LOG_LEVEL=DEBUG

# Load necessary modules
module load cuda12.6/toolkit

# Define master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Using MASTER_ADDR=$MASTER_ADDR and MASTER_PORT=$MASTER_PORT"

srun python \
  bert.py
