#!/bin/bash

#SBATCH
#SBATCH --job-name=2020050200
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 32      # cores requested
#SBATCH --mem=102400  # memory in Mb
#SBATCH -o logs/slurm_output_2020050200_job%j.out  # send stdout to outfile
#SBATCH -e logs/slurm_errfile_2020050200_job%j.out  # send stderr to errfile
#SBATCH -t 96:00:00  # time requested in hour:minute:second



cd /home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/BEACON_Inv_python
source /home/disk/p/nd349/anaconda3/etc/profile.d/conda.sh



conda activate stilt
time python template.py $1 $2 $3