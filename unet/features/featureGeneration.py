# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-06-27 16:02:28
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-06-27 16:28:50


import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from tqdm import tqdm
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
from import_met_data import readWRF
from gaussianPlume import GaussianPlume
import matplotlib.pyplot as plt

gaussian = GaussianPlume()

footprint_location = "/home/disk/hermes/nd349/barnette/data/compressed_footprints/"
met_location = "/home/disk/hermes/data/met_data/BarnettShale_2013/wrf/MYJ_LSM/"

def get_files(directory, extension=""):
    '''Return files in a given directory for given extension'''
    try:
        if extension:
            files = [f for f in listdir(directory) if f[-len(extension):] == extension]
        else:
            files = [f for f in listdir(directory)]
        files = [directory+file for file in files]
        return files
    except:
        return []

# footprint_files = get_files(footprint_location)

def checkDomain(plume, foot_lons, foot_lats, receptor_lon, receptor_lat, domain_diff=0.001):
    if abs(np.min(foot_lons)-receptor_lon)>domain_diff and abs(np.max(foot_lons)-receptor_lon)>domain_diff:
        if abs(np.min(foot_lats)-receptor_lat)>domain_diff and abs(np.max(foot_lats)-receptor_lat)>domain_diff:
            return True
    return False

def extract_feature_target(file, domain_diff=0.001, transform=''):
    # for file in footprint_files[0:]:
    # print(file)
    [directory, __, _, receptor_lat, receptor_lon, yyyy, mm, dd, hh] = file.replace('.nc', '').split("_")
    receptor_lat = float(receptor_lat)
    receptor_lon = -float(receptor_lon)
    timestamp = '-'.join(val for val in [yyyy, mm, dd]) +'_'+hh + ":00:00"
    met_file = f"{met_location}wrfout_d04_{timestamp}"

    # footprints
    foot_nc = Dataset(file)
    foot = np.array(foot_nc['foot'])
    target = foot
    foot_lats = np.array(foot_nc['lat'])
    foot_lons = np.array(foot_nc['lon'])

    # meteorology
    met_nc = readWRF(met_file)
    met_dict = vars(met_nc)
    variables = list(met_dict.keys())
    variables = [val for val in variables if val not in ['lat', 'lon']]
    # variables = ['uwind', 'vwind', 'temp', 'pblh', 'psurf']
    u_avg = -1*np.average(met_nc.uwind) # Opposite direction of the wind
    v_avg = -1*np.average(met_nc.vwind) # Opposite direction of the wind
    plume = gaussian.GaussianPlume(foot_lons, foot_lats, receptor_lon, receptor_lat, u_avg, v_avg, aA=104, aB=213, wA=10,wB=2)
    if transform=='log':
        plume[np.where(plume==0.0)]=10**-8
    if checkDomain(plume, foot_lons, foot_lats, receptor_lon, receptor_lat, domain_diff=domain_diff):
        result = np.zeros((len(variables)+1, target.shape[0], target.shape[1]))
        result[0, :, :] = plume
        # plume_short = gaussian.GaussianPlume(foot_lons, foot_lats, receptor_lon, receptor_lat, u_avg, v_avg, aA=10, aB=21)
        # plume_short[np.where(plume_short)==0.0]=10**-8
        # result[1,:,:] = plume_short
        for idx, var in enumerate(variables):
            # print(var)
            if var not in ['lat', 'lon']:
                result[idx+1, :, :] = met_dict[var]
        return result, target
    else:
        return "", ""

# h = plt.pcolor(foot_lons, foot_lats, result[0], vmin=0, vmax=10**-2, aA=10, aB=21)
# plt.colorbar(h)
# plt.savefig("exp1.png")
# plt.close()