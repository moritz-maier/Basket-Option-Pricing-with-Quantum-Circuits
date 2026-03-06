#!/bin/bash
#
# SLURM job script for running a single configuration.
# The config ID is passed as a command-line argument when submitting the job.

#SBATCH --job-name=main                # Name of the job
#SBATCH --output=outputs/output.%j.out # Standard output file (JobID)
#SBATCH --error=errors/error.%j.err    # Error output file
#SBATCH --time=3-00:00:00              # Maximum runtime (3 days)
#SBATCH --partition=All                # Partition/queue to use
#SBATCH --nodes=1                      # Number of nodes

# Activate virtual environment
source ~/work/bachelorarbeit/venv/bin/activate

# Change to project root directory
cd ~/work/bachelorarbeit/

# Run Python module with config ID passed as argument
python3 -m configs.main --config $1