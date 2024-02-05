#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16000  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o slurm-create-interim-dataset-%j.out  # %j = job ID

conda init bash
conda activate urbano2
python create_interim_datasets.py -n 3