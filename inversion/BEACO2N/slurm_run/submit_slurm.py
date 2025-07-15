# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:37:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2023-10-07 23:57:46

import os
import datetime # ; import time
import numpy as np
import pandas as pd

mode = 'integrated_decayed'
emulator = True
experiment = 'emulator'+mode

location = 'BEACON'+experiment
start = datetime.datetime(2020, 2, 2, 0, 0)
end = datetime.datetime(2020, 5, 3, 0, 0)
hours_interval = 24 # hours
difference = (end-start).total_seconds()/3600 # hours
iteration = int(difference/hours_interval)
print("Total iterations:", iteration)

nodelist = ['h11', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']
use_nodelist = False

def create_submission_bash(date, term, node=''):
    """
        Update the submission script job.sh

        Arguments:
            date: <str>
            term: <str>
        returns:
            None
    """

    with open('job.sh', 'w') as file:
        jobname = date+term
        file.writelines("#!/bin/bash\n")
        file.writelines("\n")
        file.writelines("#SBATCH\n")
        file.writelines(f"#SBATCH --job-name={jobname}\n")
        file.writelines("#SBATCH -N 1      # nodes requested\n")
        file.writelines("#SBATCH -n 1      # tasks requested\n")
        file.writelines("#SBATCH -c 32      # cores requested\n")
        file.writelines("#SBATCH --partition=HERMES      # partition requested\n")
        if node:
            file.writelines(f"#SBATCH --nodelist={node}      # nodes requested\n")
        file.writelines("#SBATCH --mem=102400  # memory in Mb\n")
        if emulator:
            file.writelines(f"#SBATCH -o /home/disk/hermes/nd349/data/inversion/runs/logs/BEACON_distanceUNet_logepsilon_thresholde-8_MSE/{mode}/{date}.out  # send stdout to outfile\n")
        else:
            file.writelines(f"#SBATCH -o /home/disk/hermes/nd349/data/inversion/runs/logs/STILT/{mode}/{date}.out  # send stdout to outfile\n")
        # file.writelines(f"#SBATCH -e /home/disk/hermes/nd349/data/inversion/runs/logs/slurm_errfile_{jobname}.out  # send stderr to errfile\n")
        file.writelines("#SBATCH -t 96:00:00  # time requested in hour:minute:second\n")
        file.writelines("\n\n\n")
        file.writelines("cd /home/disk/p/nd349/nikhil.dadheech/pointSources/Inversion/InversionEmulator/BEACON_emulator400400/\n")
        file.writelines("source /home/disk/hermes/nd349/anaconda3/etc/profile.d/conda.sh\n")
        file.writelines("\n\n\n")
        file.writelines("conda activate torch\n")
        file.writelines("time python template.py $1 $2 $3 $4")
    file.close()

for idx in range(iteration):
    batch_start = start+datetime.timedelta(hours=hours_interval*(idx))
    batch_end = start+datetime.timedelta(hours=hours_interval*(idx+1))

    batch_start_year = str(batch_start.year)
    batch_start_month = str(batch_start.month)
    batch_start_day = str(batch_start.day)
    batch_start_hour = str(batch_start.hour)

    batch_end_year = str(batch_end.year)
    batch_end_month = str(batch_end.month)
    batch_end_day = str(batch_end.day)
    batch_end_hour = str(batch_end.hour)

    if len(batch_start_month) == 1:
        batch_start_month = '0' + batch_start_month
    if len(batch_start_day) == 1:
            batch_start_day = '0' + batch_start_day
    if len(batch_start_hour) == 1:
            batch_start_hour = '0' + batch_start_hour
    if len(batch_end_month) == 1:
        batch_end_month = '0' + batch_end_month
    if len(batch_end_day) == 1:
            batch_end_day = '0' + batch_end_day
    if len(batch_end_hour) == 1:
            batch_end_hour = '0' + batch_end_hour

    batch_start = f"{batch_start_year}{batch_start_month}{batch_start_day}{batch_start_hour}"
    batch_end = f"{batch_end_year}{batch_end_month}{batch_end_day}{batch_end_hour}"

    node = nodelist[idx%len(nodelist)]
    if use_nodelist:
        create_submission_bash(f"{batch_start}", f"{experiment}", node)
    else:
        create_submission_bash(f"{batch_start}", f"{experiment}", node='')

    print (f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
    os.system(f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
    # os command to submit the job


