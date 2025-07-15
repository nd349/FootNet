# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-06-27 18:46:16

import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import geopandas as gpd
import glob, sys
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import netCDF4 as nc
import geopandas as gpd

import shapely
from shapely import Polygon, Point
from scipy.spatial import Delaunay

from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv
import torch, time

from FootNet.unetpp_model import NestedUNet as footnet

print("sys.argv:", sys.argv)

# Obs 
obs_file = "linked_obs_foot_paths.csv"
domain_obs = "/home/disk/hermes2/nd349/data/TROPOMI/CH4/Barnett/Barnett_TROPOMI_Methane_Feb_Mar_Apr_2020.csv"
obs_tau_time = 1 # day
obs_tau_space = 50 # km
cross_validation = True
cross_validation_fraction = 0.15

print("Cross validation:", cross_validation)


# Prior error covariance parameters
ems_uncert = 250/100
tau_day = 7 # Try with 14 days
print(f"Prior tau_day: {tau_day}")
tau_week = 1
min_distance = 30 # km
tau_len = 5
hq_parallel = True
off_diag = True
hq_parallel = True


# Emissions
inventory_type = "EPA" # Option between EDF and EPA
print(f"Inventory Type: {inventory_type}")
if inventory_type == "EDF":
    inventory_path = "/home/disk/hermes2/nd349/data/emissions/Inventories/CH4/EDF/EI_ME_v1.0.gpkg"
elif inventory_type == "EPA":
    inventory_path = "/home/disk/hermes2/data/emissions/Inventories/EPA/Express_Extension_Gridded_GHGI_Methane_v2_2020.nc"

spatial_covariance_path = f"data/{inventory_type}_spatial_Sa_xy.pkl"

test = False
if not test:
    argv = sys.argv[1:]
    location = argv[2]
else:
    location = "FootNet_BARNETT_TROPOMI"

location = location + f"/{inventory_type}/"
output_directory = location
print("Output direcotry:", output_directory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Met data
hr3lon_full = np.load('../../../footnet/HRRR_lon_lat.npz')['lon']
hr3lat_full = np.load('../../../footnet/HRRR_lon_lat.npz')['lat']
hr3lon_full = (hr3lon_full+180)%360-180  # convert from 0~360 to -180~180
HRRR_DIR = '/home/disk/hermes2/nd349/data/met_data/xstilt_CONUS_data_lite/'
trimsize = 150
# HRRR_DIR = "/home/disk/hermes2/data/met_data/hrrr/"


# Transport
foot_model = "FootNet" # Option between XSTILT and FootNet
transport_max_backhours = 72
transport_max_days = 4 # transport max backhours 72 are covered in these days
if foot_model == "FootNet":
    num_workers = 8
elif foot_model == "XSTILT":
    num_workers = -1 # 4 or 8 for FootNet and -1 or variable for XSTILT

# FootNet
emulator = True
batch_size = 8
n_channels = 49
n_classes = 1
epsilon = 1e-4
footnet_hours_mode='average'
maximum_domain_trajectory = 200*np.sqrt(2)
# model = FootNet.unet_model.UNet
# foot_model = NestedUNet
device = 'cuda:1' #Option out of cuda and cpu => torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emulator_model_path = "/home/disk/hermes/nd349/footprint_unet/CONUS/XSTILT/CONUS_300k/Xmodels/XNestedUNet24h_fullrun/best_modelXNestedUNet24h_fullrun.pth"

if not test:
    mode = argv[3].replace("\r", "")
else:
    mode = "resolved"

print(f"Transport Model: {foot_model}")
print("Mode:", mode)

xres = 1/120
yres = 1/120

foot_file = "/home/disk/hermes/nd349/TransportModels/X-STILT/CONUS_sites/XSTILT_output/Domain_TROPOMI/site_32.4999999999999_-97.499999999998/out_NA_hrrr_ideal/footprints/202002011900_-97.2416666666645_32.6083333333332_X_foot.nc"
sample_data = nc.Dataset(foot_file)
clat = sample_data['lat'][:][209]
clon = sample_data['lon'][:][209]
clon_index = 209
clat_index = 209

lats = np.array(sample_data['lat'])[209-200:209+200]
lons = np.array(sample_data['lon'])[209-200:209+200]


big_lons = np.arange(-100, -95, xres)
big_lats = np.arange(30, 35, yres)

nrow = lats.shape[0]
ncol = lons.shape[0]
m = nrow*ncol

model_error = { 0:2, 1:4, 2:6, 3:8, 4:5, 5:4, 6:3, 7:3, 8:3, 9:3, 10:3, 11:3, 12:3, \
13:3, 14:4, 15:5, 16:8, 17:6, 18:4, 19:2, 20:1, 21:1, 22:1, 23:1}

# Time domain

if not test:
    start_time = datetime.datetime.strptime(argv[0], "%Y%m%d%H")
    end_time = datetime.datetime.strptime(argv[1], "%Y%m%d%H")
else:
    start_time = datetime.datetime(2020, 3, 1, 0, 0)
    end_time = datetime.datetime(2020, 3, 31, 23, 0)
print("Start time:", start_time)
print("End time:", end_time)
temporal_frequency = '1d'

buffer_days = 7
print(f"Buffer days: {buffer_days}")

if mode == 'resolved':
    m_start = start_time-datetime.timedelta(days=buffer_days)
    m_end = end_time+datetime.timedelta(days=buffer_days)
    # m_end = m_end-datetime.timedelta(hours=1)
elif mode == 'integrated' or mode == 'integrated_average' or mode == 'integrated_decayed':
    m_start = start_time-datetime.timedelta(days=buffer_days)
    m_end = end_time+datetime.timedelta(days=buffer_days)
    # m_end = m_end-datetime.timedelta(hours=1)
else:
    raise Exception(f"mode should be one of ['resolved', 'integrated', 'integrated_average', 'integrated_decayed'], instead {mode} is given...")

date_range = pd.date_range(start=m_start, end=m_end, freq=temporal_frequency)

time_dict = {}
for idx, value in enumerate(date_range):
    time_dict[value] = idx

# Background
octant_dict = {}
octant_dict[1] = [lats[-1], big_lats[-1], lons[clon_index], lons[-1]]
octant_dict[2] = [lats[clat_index], lats[-1], lons[-1], big_lons[-1]]
octant_dict[3] = [lats[0], lats[clat_index], lons[-1], big_lons[-1]]
octant_dict[4] = [big_lats[0], lats[0], lons[clon_index], lons[-1]]
octant_dict[5] = [big_lats[0], lats[0], lons[0], lons[clon_index]]
octant_dict[6] = [lats[0], lats[clat_index], big_lons[0], lons[0]]
octant_dict[7] = [lats[clat_index], lats[-1], big_lons[0], lons[0]]
octant_dict[8] = [lats[-1], big_lats[-1], lons[0], lons[clon_index]]

background_start_date = datetime.datetime(2020, 1, 30, 0)
background_end_date = datetime.datetime(2020, 5, 15, 23)
background_date_range = pd.date_range(start=background_start_date, end=background_end_date, freq='1d')
