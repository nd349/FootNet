# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:37:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-06-23 22:07:26

import datetime # ; import time
import numpy as np
import pandas as pd
import sys, os

sys.argv = sys.argv[1:]
print("sys.argv:", sys.argv)

test = False
if not test:
	job = str(sys.argv[2])
	mode = str(sys.argv[3])
else:
	job = ''
	mode = 'integrated_decayed'

emulator = True
emulator_run = False

location = job.upper()
back_hours = 36
diag_prior = False
full_prior = True
sparse = True
device = 'cpu'
hq_parallel = True
cross_validation = True
cross_validation_fraction = 0.15

# Directories and files
footprint_directory = "/home/disk/hermes/data/footprints/BEACO2N/obs/"
# emulator_file_path = "/home/disk/hermes/taihe/footnet_test/noAD850/emulatorMSE/"
emulator_file_path = "/home/disk/hermes2/nd349/data/footprints/NestedUNet24h_fullrun/"
emission_directory = "/home/disk/hermes/data/emissions/BEACO2N/"
# emulator_model_file = '/data/Unet_checkpt_0.58_mixed.h5'

if emulator:
	output_directory = f"/home/disk/hermes/nd349/data/inversion/posterior/BEACON/FootNet_off_diag_Sa_xy/NestedUNet24h_fullrun/{mode}/"
else:
	output_directory = f"/home/disk/hermes/nd349/data/inversion/posterior/BEACON/STILT_off_diag_Sa_xy/{mode}/"

if not os.path.exists(output_directory):
	raise Exception(f"The output_directory: {output_directory} does not exist ....")
else:
	pass # You can also make the output directory here if you want!
# print(f"Output directory: {output_directory}:")

# Model error
# mod_err = [[00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23] # hour in UTC
#       [ 2 4 6 8 5 4 3 3 3 3 3 3 3 3 4 5 8 6 4 2 1 1 1 1]] # error (ppm)

model_error = { 0:2, 1:4, 2:6, 3:8, 4:5, 5:4, 6:3, 7:3, 8:3, 9:3, 10:3, 11:3, 12:3, \
13:3, 14:4, 15:5, 16:8, 17:6, 18:4, 19:2, 20:1, 21:1, 22:1, 23:1}

# Uncertainties
ems_uncert = 50/100
minUncert = 1.0
tau_day = 1
tau_hr = 5
tau_len = 5
tau_time = 1 # hour
tau_space = 2 # km
mVal = minUncert/ems_uncert
lowBound = 1e-5
obsFreq    = 60.0    # Observations per hour
fsigma=0.5

### Different grids to use
# Full grid used for footprints
full_xLim = [ -125.0, -120.0 ]
full_yLim = [   36.0,   40.0 ]
big_xLim,big_yLim = full_xLim,full_yLim
num_lats = 481
num_lons = 601

# Grid
# lats = np.linspace(full_yLim[0], full_yLim[1], num_lats, dtype=np.float32)
# lons = np.linspace(full_xLim[0], full_xLim[1], num_lons, dtype=np.float32)

orig_lats = np.linspace(full_yLim[0], full_yLim[1], num_lats)
orig_lons = np.linspace(full_xLim[0], full_xLim[1], num_lons)

clon_index = int(orig_lons.shape[0]/2)
clat_index = int(orig_lats.shape[0]/2)
clon = orig_lats[clon_index]
clat = orig_lons[clat_index]
lats = orig_lats[clat_index-200:clat_index+200]
lons = orig_lons[clon_index-200:clon_index+200]

nrow = 400
ncol = 400
m = nrow*ncol

# Medium sized grid
medium_xLim = [-123.60,-121.60]
medium_yLim = [  36.80,  38.60]
med_xLim,med_yLim = medium_xLim,medium_yLim
# Bay Area domain (smallest grid)
BayArea_xLim = [-123.10,-121.80]
BayArea_yLim = [  37.35,  38.40]
small_xLim,small_yLim = BayArea_xLim,BayArea_yLim
# Inversion grid to use
Inv_lonLim = small_xLim
Inv_latLim = small_yLim


# Time domain
if test:
	print("job field is empty:", sys.argv)
	start_time = datetime.datetime(2020, 2, 2, 0, 0)
	end_time = datetime.datetime(2020, 2, 3, 0, 0)

else:
	start_time = sys.argv[0] #yyyymmddhh (str)
	end_time = sys.argv[1] #yyyymmddhh (str)
	if len(start_time) !=10 or len(end_time)!=10:
		raise Exception(f"The expected format for the start and end date is yyyymmddhh but {start_time} {end_time} is given")
	else:
		start_time = datetime.datetime(int(start_time[0:4]), int(start_time[4:6]), int(start_time[6:8]), int(start_time[8:10]))
		end_time = datetime.datetime(int(end_time[0:4]), int(end_time[4:6]), int(end_time[6:8]), int(end_time[8:10]))

if mode == 'resolved':
	m_start = start_time-datetime.timedelta(hours=back_hours)
	m_end = end_time+datetime.timedelta(hours=back_hours)
	m_end = m_end-datetime.timedelta(hours=1)
elif mode == 'integrated' or mode == 'integrated_average' or mode == 'integrated_decayed':
	m_start = start_time-datetime.timedelta(hours=back_hours)
	m_end = end_time+datetime.timedelta(hours=back_hours)
	m_end = m_end-datetime.timedelta(hours=1)


date_range = pd.date_range(start=m_start, end=m_end, freq='1h')

time_dict = {}
for idx, value in enumerate(date_range):
    time_dict[value] = idx

## Emulator parameters
HRR_lon_lat_npz = "data/HRRR_lon_lat.npz"



