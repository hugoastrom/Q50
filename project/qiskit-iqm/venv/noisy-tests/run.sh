#!/bin/bash

#SBATCH --job-name=quantumjob        # Job name
#SBATCH --account=project_462001017  # Project for billing (slurm_job_account)
#SBATCH --partition=q_fiqci          # Partition (queue) name
#SBATCH --ntasks=1                   # One task (process)
#SBATCH --mem-per-cpu=1G             # memory allocation
#SBATCH --cpus-per-task=1            # Number of cores (threads)
#SBATCH --time=00:20:00              # Run time (hh:mm:ss)

module use /appl/local/quantum/modulefiles
module load fiqci-vtt-qiskit/

python -m pip install --user pyscf
python -m pip install --user numpy
python -m pip install --user scipy
python -m pip install --user qiskit_nature

export DEVICES=("Q50")
source $RUN_SETUP

python main.py "He 0.0 0.0 0.0" "3-21g" 0 "cobyla" "backend_estimator"
