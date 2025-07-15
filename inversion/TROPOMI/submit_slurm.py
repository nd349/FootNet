# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:37:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-06-25 18:55:39

import os
from os import listdir
import datetime # ; import time
import numpy as np
import pandas as pd
import random
import sys, glob

mode = 'resolved'
emulator = True
# if emulator:
#     experiment = 'EPA_emulator_'+mode
# else:
#     experiment = 'EPA_XSTILT_'+mode
experiment = f"XNestedUNet24h_fullrun_month"
# experiment = f"XSTILT"

experiment += f"_{mode}"
output_dir = "/home/disk/hermes3/nd349/data/inversion/TROPOMI/Barnett/"
location = output_dir+'BARNETT_'+experiment

if not os.path.exists(location):
    os.makedirs(location)

start = datetime.datetime(2020, 2, 2, 0, 0)
end = datetime.datetime(2020, 5, 2, 0, 0)
temporal_frequency = '7d'

date_range = pd.date_range(start=start, end=end, freq=temporal_frequency)
time_res = date_range[1] - date_range[0]


timestamps = [datetime.datetime.strftime(val, '%Y%m%d%H')[:8] for val in date_range]

timestamps = set([f"{val[:4]}x{val[4:6]}x{val[6:]}" for val in timestamps])
# print(timestamps, len(timestamps))

print(date_range)
print("Timestamps:", timestamps)

files = glob.glob(location + "/*.ncdf")

files = set([val.split("_")[-1].replace(".ncdf", "") for val in files])
remaining_timestamps = list(timestamps - files.intersection(timestamps))
remaining_timestamps.sort()
remaining_timestamps = [val.replace("x", "")+"00" for val in remaining_timestamps]
remaining_timestamps = [(datetime.datetime.strptime(val, '%Y%m%d%H'), datetime.datetime.strptime(val, '%Y%m%d%H') + time_res - datetime.timedelta(hours=1)) for val in remaining_timestamps]
remaining_timestamps = [(datetime.datetime.strftime(val[0], '%Y%m%d%H'), datetime.datetime.strftime(val[1], '%Y%m%d%H')) for val in remaining_timestamps]

print("Remaining_timestamps:", remaining_timestamps)
# import pdb; pdb.set_trace()
# print(a)

run_model = True
slurm = True


nodelist = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h8', 'h10', 'h11', 'h12']
hermes_list = ['h'+str(i) for i in range(1, 11)]
debug_list = ['h11', 'h12']
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
        if node:
            if node in debug_list:
                file.writelines(f"#SBATCH --partition=DEBUG\n")
            else:
                file.writelines(f"#SBATCH --partition=HERMES\n")
        else:
            file.writelines(f"#SBATCH --partition=GPU\n")
        file.writelines("#SBATCH -N 1      # nodes requested\n")
        file.writelines("#SBATCH -n 32      # tasks requested\n")
        file.writelines("#SBATCH -c 1      # cores requested\n")
        if node:
            file.writelines(f"#SBATCH --nodelist={node}      # nodes requested\n")
        file.writelines("#SBATCH --mem=102400  # memory in Mb\n")
        # if emulator:
        file.writelines(f"#SBATCH -o {location}/{date}.out  # send stdout to outfile\n")
        # else:
        #     file.writelines(f"#SBATCH -o /home/disk/hermes/nd349/data/inversion/runs/logs/STILT/{mode}/{date}.out  # send stdout to outfile\n")
        # file.writelines(f"#SBATCH -e /home/disk/hermes/nd349/data/inversion/runs/logs/slurm_errfile_{jobname}.out  # send stderr to errfile\n")
        file.writelines("#SBATCH -t 24:00:00  # time requested in hour:minute:second\n")
        file.writelines("\n\n\n")
        file.writelines("cd /home/disk/hermes/nd349/nikhil.dadheech/pointSources/Inversion/InversionEmulator/TROPOMI/\n")
        file.writelines("source /home/disk/hermes/nd349/anaconda3/etc/profile.d/conda.sh\n")
        file.writelines("\n\n\n")
        file.writelines("conda activate torch\n")
        file.writelines("time python main.py $1 $2 $3 $4")
    file.close()

for idx, val in enumerate(remaining_timestamps):
    print(idx, val)
    batch_start = val[0]
    batch_end = val[1]
    node = nodelist[idx%len(nodelist)]
    if run_model:
        if slurm:
            if use_nodelist:
                create_submission_bash(f"{batch_start}", f"{experiment}", node)
            else:
                create_submission_bash(f"{batch_start}", f"{experiment}", node='')

            print (f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
            os.system(f"sbatch job.sh {batch_start} {batch_end} {location} {mode}")
        
        else:
            os.system(f"time python template.py {batch_start} {batch_end} {location} {mode}")