import pandas as pd
import numpy as np
import glob,random, time
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import num2date,date2num
from tqdm import tqdm
import xarray as xr
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
from math import asin, atan2, cos, degrees, radians, sin
from joblib import Parallel, delayed


year = 2020
list_files = glob.glob(f"PATH TO HRRR FILES*")
print(f"Total files in {year}: {len(list_files)}")
print(list_files[:10])
output_file_path = "PATH"
# for file in tqdm(list_files):
def get_trim_data(file):
    """
    This function loads HRRR original files and create corresponding lite netcdf file with met fields which are required for the FootNet model.
    file: <str>
    """
    # Load HRRR original files using pseudonetcdf engine
    fh = xr.open_dataset(file, engine='pseudonetcdf')
    
    outfile = f'{output_file_path}/{year}/{file.split("/")[-1].replace(".hrrra", ".nc")}'
    out_nc = nc.Dataset(outfile, "w", format="NETCDF4")
    out_nc.createDimension("x", fh['x'].shape[0])
    out_nc.createDimension("y", fh['y'].shape[0])
    out_nc.createDimension("time", fh['time'].shape[0])
    
    out_nc.createVariable("time", np.int64, ("time"))[:] = [int(val) for val in np.array(fh['time'])]
    
    out_nc.createVariable("U10M", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['U10M'])
    out_nc.createVariable("V10M", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['V10M'])
    out_nc.createVariable("PBLH", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['PBLH'])
    out_nc.createVariable("PRSS", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['PRSS'])
    
    out_nc.createVariable("UWND9_850hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['UWND'][:, 9, :, :])
    out_nc.createVariable("VWND9_850hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['VWND'][:, 9, :, :])
    
    out_nc.createVariable("UWND17_500hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['UWND'][:, 17, :, :])
    out_nc.createVariable("VWND17_500hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['VWND'][:, 17, :, :])
    
    out_nc.createVariable("PRES9_850hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['PRES'][:, 9, :, :])
    out_nc.createVariable("TEMP9_850hPa", np.float32, ("time", "y", "x"))[:, :, :] = np.array(fh['TEMP'][:, 9, :, :])
    
    out_nc.close()
    fh.close()

OUTPUT = Parallel(n_jobs=4, verbose=0, backend='multiprocessing')(delayed(get_trim_data)(file) for file in tqdm(list_files))
