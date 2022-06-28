# -*- coding: utf-8 -*-
# @Author: nikhildadheech and alexturner
# @Date:   2022-06-19 19:48:06
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-06-28 13:33:22


from barnette_regridding import regridData
import time
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed


# compressed_footprint
foot_file = '/home/disk/hermes/nd349/barnette/data/compressed_footprints/compressed_foot_31.5_96.4_2013_10_19_08.nc'
footprintData = nc.Dataset(foot_file)
footLats = np.array(footprintData['lat'])
footLons = np.array(footprintData['lon'])

class MET(object):
   def __init__(self, lon, lat, press, psurf, hgt, temp, rhfr, uwind, vwind, wwind, pblh, qfx, smois, tke):
      self.lon   = lon
      self.lat   = lat
      self.press = press
      self.psurf = psurf
      self.hgt   = hgt
      self.temp  = temp
      self.rhfr  = rhfr
      self.uwind = uwind
      self.vwind = vwind
      self.wwind = wwind
      self.pblh  = pblh
      self.qfx   = qfx
      self.smois = smois
      self.tke   = tke

### Load data from our WRF met file
def qair2rh(qair, temp, press):
   # Convert temp & pressure
   temp  = temp + 273.15
   press = press * 100
   # End AJT convert
   es =  6.112 * np.exp((17.67 * temp)/(temp + 243.5))
   e  = qair * press / (0.378 * qair + 0.622)
   rh = e / es
   rh[rh > 1] <- 1
   rh[rh < 0] <- 0
   return rh

def readWRF(fname, layers=3):
    # Open the file
    fh = Dataset(fname, mode='r')
    # Get data about this observation
    lon   = fh.variables['XLONG'][:]                    # Longitude (deg north)
    lat   = fh.variables['XLAT'][:]                     # Latitude (deg east)
    hgt   = np.array(fh.variables['HGT'][0,:,:])     # Terrain height (m)
    uwind = np.array(fh.variables['U10'][0,:,:])  #fh.variables['U'][:]   # x-wind component (m/s)
    vwind = np.array(fh.variables['V10'][0,:,:])  #fh.variables['V'][:]    # y-wind component (m/s)
    pblh  = np.array(fh.variables['PBLH'][0,:,:]) # PBL height (m)
    qfx   = np.array(fh.variables['QFX'][0,:,:])  # Upward moisture flux at the surface
    psurf = np.array(fh.variables['PSFC'][0,:,:])                     # Surface pressure (Pa)

    
    # multiple layer data
    wwind = np.array(fh.variables['W'][0,:,:])    # z-wind component (m/s)
    qv    = np.array(fh.variables['QVAPOR'][0,:,:])    # Water vapor mixing ratio (kg/kg)
    smois = np.array(fh.variables['SMOIS'][0,:,:])  # Soil moisture (m3/m3)
    tke   = np.array(fh.variables['TKE_PBL'][0,:,:])  # TKE (m2/s2)
    press = np.array(fh.variables['P'][0,:,:]) + np.array(fh.variables['PB'][0,:,:])   # Pressure (Pa)
    pot = np.array(fh.variables['T'][0,:,:]) + np.array(fh.variables['T00'])
    temp  = pot*(press/psurf)**0.2854
    rhfr  = qair2rh(qv,temp,press)          # Relative humidity (fraction)
    
    # averaging over layers
    wwind = regridData(np.sum(wwind[:layers, :, :], axis=0)/layers, footLats, footLons)
    qv = regridData(np.sum(qv[:layers, :, :], axis=0)/layers, footLats, footLons)
    smois = regridData(np.sum(smois[:layers, :, :], axis=0)/layers, footLats, footLons)
    tke = regridData(np.sum(tke[:layers, :, :], axis=0)/layers, footLats, footLons)
    press = regridData(np.sum(press[:layers, :, :], axis=0)/layers, footLats, footLons)
    temp = regridData(np.sum(temp[:layers, :, :], axis=0)/layers, footLats, footLons)
    rhfr = regridData(np.sum(rhfr[:layers, :, :], axis=0)/layers, footLats, footLons)

    hgt = regridData(hgt,footLats, footLons)
    uwind = regridData(uwind,footLats, footLons)
    vwind = regridData(vwind,footLats, footLons)
    pblh = regridData(pblh,footLats, footLons)
    qfx = regridData(qfx,footLats, footLons)
    psurf = regridData(psurf,footLats, footLons)
    
    # Close the file
    fh.close()
    # Make the object and return it
    met = MET(lon,lat,press,psurf,hgt,temp,rhfr,uwind,vwind,wwind,pblh,qfx,smois,tke)
    return met


if __name__ == '__main__':
	met = readWRF("/home/disk/hermes/data/met_data/BarnettShale_2013/wrf/MYJ_LSM/wrfout_d04_2013-11-01_00:00:00")
	print(met.temp)


