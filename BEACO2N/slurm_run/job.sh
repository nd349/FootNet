#!/bin/bash

#SBATCH
#SBATCH --job-name=2020050200emulatorintegrated_decayed
#SBATCH --partition=HERMES
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 32      # cores requested
#SBATCH --mem=102400  # memory in Mb
#SBATCH -o /home/disk/hermes/nd349/data/inversion/runs/logs/FootNet_off_diag_Sa_xy/NestedUNet24h_fullrun/integrated_decayed/2020050200.out  # send stdout to outfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second



cd /home/disk/hermes/nd349/nikhil.dadheech/pointSources/Inversion/InversionEmulator/BEACON_emulator400400/
source /home/disk/hermes/nd349/anaconda3/etc/profile.d/conda.sh



conda activate torch
time python template.py $1 $2 $3 $4