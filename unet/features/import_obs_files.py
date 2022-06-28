# -*- coding: utf-8 -*-
# @Author: nikhildadheech and alexturner
# @Date:   2022-06-14 09:19:55
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-06-28 13:33:31

from netCDF4 import Dataset
# from mpl_toolkits import basemap
import numpy as np
import os
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


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

class OBS(object):
    def __init__(self, lon, lat, nhrs, tcld, foot, receptor_lat, receptor_lon, timestamp):
        self.lon  = lon
        self.lat  = lat
        self.nhrs = nhrs
        self.tcld = tcld
        self.foot = foot
        self.receptor_lat = receptor_lat
        self.receptor_lon = receptor_lon
        self.timestamp = timestamp
    

### Load data from our footprint file
def readOBS(fname, receptor_lat, receptor_lon, timestamp):
    # Open the file
    fh = Dataset(fname, mode='r')
    # Get data about this observation
    lon  = fh.variables['footnflon'][:]
    lat  = fh.variables['footnflat'][:]
    nhrs = fh.variables['nfootnfhrs'][:]
    tcld = fh.variables['finfo_tcld'][:]
    #foot = fh.variables['footnf_surface'][:]
    #foot = fh.variables['footnf_CH4Sat'][:]
    foot = np.sum(fh.variables['footnf_surface'][:],axis=0)
    # foot = np.sum(fh.variables['footnf_CH4Sat'][:],axis=0)
    # Close the file
    fh.close()
    # Make the object and return it
    obs = OBS(lon,lat,nhrs,tcld,foot, receptor_lat, receptor_lon, timestamp)
    return obs

def writeOBS(out_file, obs):
    out_nc = Dataset(out_file, 'w', format='NETCDF4')
    out_nc.createDimension("lat", obs.lat.shape[0])
    out_nc.createDimension("lon", obs.lon.shape[0])
    out_nc.createDimension("info", 1)
    lat = out_nc.createVariable("lat", 'f8', ("lat",))
    lon = out_nc.createVariable("lon", 'f8', ("lon",))
    out_foot = out_nc.createVariable("foot", 'f8', ("lat", "lon"))
    lat[:] = obs.lat
    lon[:] = obs.lon
    out_foot[:,:] = obs.foot
    out_nc.createVariable("receptor_latitutde", "f8", ("info"))[:] = obs.receptor_lat
    out_nc.createVariable("receptor_longitude", "f8", ("info"))[:] = obs.receptor_lon
    out_nc.createVariable("timestamp", "f8", ("info"))[:] = obs.timestamp
    out_nc.close()

def get_obs_data(obs_dir, YYYY, MM, DD, HH, write=False):
    for yyyy in YYYY:
        for mm in MM:
            for dd in DD:
                print("timestamp:", yyyy,mm,dd)
                for hh in HH:
                    bDir = '%s/%04i/%02i/%02i/%02i/' % (obs_dir,yyyy,mm,dd,hh)
                    observation_files = get_files(bDir)
                    if observation_files:
                        try:
                            for file in tqdm(observation_files):
                                [_, lat, lon, timestamp] = file.split("/")[-1].replace("Nx", "_").replace("Wx", "_").split("_")
                                receptor_lat = float(lat)
                                receptor_lon = float(lon)
                                timestamp = timestamp.replace('x', '_').replace('.nc', '')
                                data = readOBS(file, receptor_lat, receptor_lon, timestamp)
                                if write:
                                    out_file = f"{output_comp_foot}compressed_foot_{receptor_lat}_{receptor_lon}_{timestamp}.nc"
                                    writeOBS(out_file, data)
                        except Exception as e:
                            print(e, "error a")

if __name__ == '__main__':
    baseDir = '/home/disk/hermes/data'
    obs_dir  = baseDir+'/footprints/BarnettShale_2013'
    YYYY     = [2013]
    MM       = [10]
    # DD       = list(range(19,26))
    DD       = list(range(1,31))
    # HH       = list(range(0,24))
    HH       = list(range(00,23))
    output_comp_foot = "/home/disk/hermes/nd349/barnette/data/compressed_footprints/"
    get_obs_data(obs_dir, YYYY, MM, DD, HH)
