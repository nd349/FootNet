# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-06-23 16:28:40
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-06-27 16:27:42


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from tqdm import tqdm
from joblib import Parallel, delayed

# compressed_footprint_example
foot_file = '/home/disk/hermes/nd349/barnette/data/compressed_footprints/compressed_foot_31.5_96.4_2013_10_19_08.nc'
footprintData = nc.Dataset(foot_file)
footLats = np.array(footprintData['lat'])
footLons = np.array(footprintData['lon'])

# Met data_example
met_file = "/home/disk/hermes/data/met_data/BarnettShale_2013/wrf/MYJ_LSM/wrfout_d04_2013-11-01_00:00:00"
met_data = nc.Dataset(met_file)
met_lats = met_data['XLAT'][0,:,:]
met_lons = met_data['XLONG'][0,:,:]

#output_file_location
outputfile="../data/regrid_barnette.csv"

def getClosestLambertianProjection(lats, lons, receptorLat, receptorLon):
    c = np.sqrt((lats-receptorLat)**2+(lons-receptorLon)**2)
    result = np.where(c==c.min())
    return (result[0][0], result[1][0])

def getNearestGrids(met_lats, met_lons, footLons, lat):
    regridList = []
    for jdx in range(footLons.shape[0]):
        nearestGrid = getClosestLambertianProjection(met_lats, met_lons, lat, footLons[jdx])
        regridList.append([lat, footLons[jdx], nearestGrid[0], nearestGrid[1]])
    return regridList

def getIndices(outputfile):

	regridList = Parallel(n_jobs=128, verbose=10000, backend='multiprocessing')(delayed(getNearestGrids)(met_lats, met_lons,footLons, lat) for lat in footLats)

	finalregrid = []
	for value in regridList:
	    finalregrid += value
	regrid_df = pd.DataFrame(finalregrid, columns=['lat', 'lon', 'row', 'column'])
	regrid_df['combined'] = [(regrid_df['row'][idx], regrid_df['column'][idx]) for idx in range(regrid_df.shape[0])]
	regrid_df.to_csv(outputfile, index=False)
	return

def readRegridData(file=outputfile):
	regrid_df = pd.read_csv(file)
	regrid_df['combined'] = [(regrid_df['row'][idx], regrid_df['column'][idx]) for idx in range(regrid_df.shape[0])]
	return regrid_df

regrid_df = readRegridData()

def regridData(data, lats, lons):
	originalData = np.array(data)
	transformedData = np.zeros((lats.shape[0], lons.shape[0]))
	count = 0
	for idx in range(transformedData.shape[0]):
		for jdx in range(transformedData.shape[1]):
			transformedData[idx][jdx] = originalData[regrid_df['combined'][count]]
			count += 1
	return transformedData

if __name__ == '__main__':
	# getIndices(outputfile)
	# regrid_df = readRegridData()
	start = time.time()
	regridData(met_data['U10'][0,:,:], footLats, footLons)
	print(time.time()-start)

