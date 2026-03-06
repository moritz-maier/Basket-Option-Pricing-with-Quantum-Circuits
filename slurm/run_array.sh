#!/bin/bash
#
# SLURM job script for running multiple configurations via array jobs.
# Each task in the array receives a different config ID via SLURM_ARRAY_TASK_ID.

#SBATCH --job-name=main                  # Name of the job
#SBATCH --output=outputs/output.%A_%a.out  # Standard output file (JobID_ArrayID)
#SBATCH --error=errors/error.%A_%a.err     # Error output file
#SBATCH --time=3-00:00:00                # Maximum runtime (3 days)
#SBATCH --partition=All                 # Partition/queue to use
#SBATCH --nodes=1                       # Number of nodes

# Activate virtual environment
source ~/work/bachelorarbeit/venv/bin/activate

# Change to project root directory
cd ~/work/bachelorarbeit/

# Run Python module with configuration determined by array task ID
python3 -m configs.main --config $SLURM_ARRAY_TASK_ID