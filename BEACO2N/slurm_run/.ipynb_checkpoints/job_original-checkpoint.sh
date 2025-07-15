#!/bin/bash

#SBATCH 
#SBATCH --job-name=$1
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 32      # cores requested
#SBATCH --mem=102400  # memory in Mb
#SBATCH -o slurm_output_%j.out  # send stdout to outfile
#SBATCH -e slurm_errfile_%j.out  # send stderr to errfile
#SBATCH -t 96:00:00  # time requested in hour:minute:second



# Incorporate timescale in the job name to monitor them and submit the slurm through array in sbatch (assigns same job id)
cd /home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/BEACON_Inv_python
source /home/disk/p/nd349/anaconda3/etc/profile.d/conda.sh

# Run python script
conda activate stilt

# python script.py 900000000
python template.py $1 $2 $3