# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-22 11:33:45
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 14:51:35

import sys
import datetime; import time
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import netCDF4 as nc
from Utils.readData  import *
from Utils.filter_query import *
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy import sparse
from scipy.sparse.linalg import inv
import geopy.distance; import pickle

def flatten_2d_column(foot):
    sub_foot = np.zeros((foot.shape[0]*foot.shape[1]))
    for idx in range(foot.shape[1]):
        sub_foot[idx*foot.shape[0]:(idx+1)*foot.shape[0]] = foot[:, idx]
    return sub_foot
    
# footprint_directory = "/home/disk/hermes/data/footprints/BEACO2N/obs/"
emission_directory = "/home/disk/hermes/data/emissions/BEACO2N/"

# footprint_files = get_files(footprint_directory)
emission_files = get_files(emission_directory)
emission_files.sort()
# footprint_files.sort()
print(len(emission_files))

### Different grids to use
# Full grid used for footprints
full_xLim = [ -125.0, -120.0 ]
full_yLim = [   36.0,   40.0 ]
big_xLim,big_yLim = full_xLim,full_yLim
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
lowBound = 1e-5
fsigma = 0.5

# Number of lats and lons in the original data
num_lats = 481
num_lons = 601

# Number of lats and lons in the emulator output
nrow = 400
ncol = 400

lats = np.linspace(full_yLim[0], full_yLim[1], num_lats)
lons = np.linspace(full_xLim[0], full_xLim[1], num_lons)
orig_lats = lats
orig_lons = lons

clon_index = int(lons.shape[0]/2)
clat_index = int(lats.shape[0]/2)
clon = lats[clon_index]
clat = lons[clat_index]
lats = lats[clat_index-200:clat_index+200]
lons = lons[clon_index-200:clon_index+200]

m = nrow*ncol

start_time = datetime.datetime(2020, 3, 15, 0, 0)
end_time = datetime.datetime(2020, 3, 22, 0, 0)

m_start = start_time-datetime.timedelta(hours=71)
m_end = end_time-datetime.timedelta(hours=1)
date_range = pd.date_range(start=m_start, end=m_end, freq='1h')
time_dict = {}
for idx, value in enumerate(date_range):
    time_dict[value] = idx

emission_filtered_files, emission_files_df = filter_emissions(emission_files, [start_time-datetime.timedelta(hours=71), end_time])
# foot_filtered_files, foot_files_df = filter_obs(footprint_files, Inv_lonLim, Inv_latLim, agl_domain='')

X = np.zeros((date_range.shape[0]*m, 1))
grid_flattened = [(lat, lon) for lon in lons for lat in lats]
for ems_file in tqdm(emission_filtered_files):
    trimmed_file = ems_file.split("/")[-1].replace("_", "x")
    [_, year, month, day, hour] = trimmed_file.replace('.ncdf', '').split("x")
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    timestamp = datetime.datetime(year, month, day, hour)
    ems_data = np.array(nc.Dataset(ems_file)['flx_total'])[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
    ems_flattened = flatten_2d_column(ems_data)
    index = time_dict[timestamp]
    X[index*m:(index+1)*m, 0] = ems_flattened


def if_ocean(column):
    if not np.any(column):
        return True
    else:
        return False

tau_len = 5

def get_distance(coords_1, coords_2):
    return geopy.distance.geodesic(coords_1, coords_2).km

def get_len(coords_1, coords_2):
    lat1 = coords_1[0]*np.pi/180
    lon1 = coords_1[1]*np.pi/180
    lat2 = coords_2[0]*np.pi/180
    lon2 = coords_2[1]*np.pi/180
    R = 6371e3
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # d = R ⋅ c
    a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon1-lon2)/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c/1000 #km


# def build_xy(tau_len, x_pri):

nEms = int(X.shape[0]/m)
nG = m
min_distance = 30

emsAll = np.zeros((nEms, nG), dtype=np.float32)
variance_xy = np.zeros((nG, 1), dtype=np.float32)
# dist_mat = np.zeros((nG, nG))
# dist_mat = csr_matrix()
ocean = np.zeros((nG), dtype=np.float32)
# Sa_xy = np.zeros((nG, nG))

for i in range(nG):
    emsAll[:, i] = np.array([X[j*m:(j+1)*m, 0][i] for j in range(nEms)])
    variance_xy[i, 0] = np.var(emsAll[:, i])
    if if_ocean(emsAll[:, i]):
        ocean[i] = 1

variance_xy_rows = [i for i in range(nG)]
variance_xy = csr_matrix((variance_xy[:, 0], (variance_xy_rows, variance_xy_rows)), 
                          shape = (nG, nG), dtype=np.float32)

Sa_xy = csr_matrix((nG, nG))

batch_size = 128
range_list = np.array([(i, i+batch_size) for i in np.arange(0, nG+batch_size, batch_size)])
# range_list = [(i, min(i+128, 289160)) for i in np.arange(0, 289160, 128)]
# range_list.reverse()

def parallel_fill_Sa_xy(irange, emsAll, grid_flattened, ocean):
    rows_saxy = []
    cols_saxy = []
    data_saxy = []
    min_distance = 30
    tau_len = 5
    for i in range(irange[0], irange[1]):
        for j in range(i, nG):
            cor_val = 0
            found = True
            if ocean[i] and not ocean[j]:
                found = False
            elif ocean[j] and not ocean[i]:
                found = False
            sig_val = 0
            distance = get_len(grid_flattened[i], grid_flattened[j])
            if distance < min_distance:
                if ocean[i] and ocean[j]:
                    cor_val = 1.0
                elif not ocean[i] and not ocean[j]:
                    if nEms>1:
                        cor_val = np.corrcoef(emsAll[:, i], emsAll[:, j])[0, 1]
                    else:
                        cor_val = 1.0
                dist_decay = np.exp(-abs(distance)/tau_len)
                sig_val = cor_val*dist_decay
                if sig_val > lowBound and found:
                    rows_saxy.append(i)
                    cols_saxy.append(j)
                    rows_saxy.append(j)
                    cols_saxy.append(i)
                    data_saxy.append(sig_val)
                    data_saxy.append(sig_val)
                    # Sa_xy[i, j] = sig_val
                    # Sa_xy[j, i] = sig_val
    return rows_saxy, cols_saxy, data_saxy

# start_index = int(sys.argv[1])
# print(start_index)
# interval = int(sys.argv[2])
OUTPUT = Parallel(n_jobs=16, verbose=1000, backend='multiprocessing')(delayed(parallel_fill_Sa_xy)(i, emsAll, grid_flattened, ocean) for i in tqdm(range_list))

rows = []
cols = []
data = []
for value in OUTPUT:
    rows += value[0]
    cols += value[1]
    data += value[2]

Sa_xy = csr_matrix((data, (rows, cols)), 
                          shape = (nG, nG), dtype=np.float32)

#Sa_xy = np.sqrt(fsigma)*(csr_matrix.dot(csr_matrix.sqrt(variance_xy), csr_matrix.dot(Sa_xy, csr_matrix.sqrt(variance_xy))))
Sa_xy = csc_matrix(Sa_xy)
print("Shape of the Sa_xy matrix:", Sa_xy.shape)
with open(f"data/Sa_xy_corrcoef_emulator.pkl", "wb") as file:
    pickle.dump(Sa_xy, file)
