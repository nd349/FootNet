import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import os, re
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import h5py, multiprocessing, time

inputPath = "/home/disk/hermes/nd349/out/footprints/"
outputPath = "/home/disk/hermes/nd349/out/compressedFootprints/"

total_footprints = [f for f in listdir(inputPath) if f[-3:] == '.nc']
existingCompressedFootprints = [f for f in listdir(outputPath) if f[-3:]==".nc"]
footprints = [f for f in total_footprints if "compressed_"+f not in existingCompressedFootprints]
print(len(footprints), len(existingCompressedFootprints), len(total_footprints))

print("inputPath:", inputPath)
print("outputPath:", outputPath)

def checkAlphaNumeric(string):
    if re.search('\d', string):
        return True
    else:
        return False

def getInfo(initial_text):
    text = initial_text.replace(".nc", "").replace("obs", "").split("_")
    info_list = []
    for term in text:
        if checkAlphaNumeric(term):
            if type(eval(term))==int or type(eval(term))==float:
                info_list.append(term)
    if len(info_list[0]) ==10:
        yyyy = info_list[0][:4]
        mm = info_list[0][4:6]
        dd = info_list[0][6:8]
        hh = info_list[0][8:]
    receptorLon = info_list[1]
    receptorLat = info_list[2]
    heightAboveGround = info_list[3]
    return yyyy, mm, dd, hh, receptorLon, receptorLat, heightAboveGround

def compressFootprint(files, inputPath=inputPath, outputPath=outputPath):
    for file in files:
        try:
            data = nc.Dataset(inputPath+file)
            inputLat = list(np.array(data['lat']))
            inputLon = list(np.array(data['lon']))
            footC = np.nansum(np.array(data['foot']), axis=0)
            yyyy, mm, dd, hh, receptorLon, receptorLat, heightAboveGround = getInfo(file)
            # plt.spy(footC)
            # plt.show()
            out_nc = nc.Dataset(outputPath+"compressed_"+file, 'w', format='NETCDF4')
            out_nc.createDimension("lat", 481)
            out_nc.createDimension("lon", 601)
            out_nc.createDimension("info", 1)
            lat = out_nc.createVariable("lat", 'f8', ("lat",))
            lon = out_nc.createVariable("lon", 'f8', ("lon",))
            out_foot = out_nc.createVariable("foot", 'f8', ("lat", "lon"))
            lat[:] = inputLat
            lon[:] = inputLon
            out_foot[:,:] = footC
            out_nc.createVariable("yyyy", "f8", ("info"))[:] = yyyy
            out_nc.createVariable("mm", "f8", ("info"))[:] = mm
            out_nc.createVariable("dd", "f8", ("info"))[:] = dd
            out_nc.createVariable("hh", "f8", ("info"))[:] = hh
            out_nc.createVariable("receptorLon", "f8", ("info"))[:] = receptorLon
            out_nc.createVariable("receptorLat", "f8", ("info"))[:] = receptorLat
            out_nc.createVariable("HAGL", "f8", ("info"))[:] = heightAboveGround
            out_nc.close()
        except Exception as e:
            print(e)
            #remove the files
            pass
            # print(fileNumber, e)
            # print(file)

if __name__ == '__main__':
    starttime = time.time()
    processes = []
    batch_size = 20
    footprintBatches = [footprints[idx*batch_size:(idx+1)*batch_size] for idx in range(len(footprints)) if footprints[idx*batch_size:(idx+1)*batch_size]]
    OUTPUT = Parallel(n_jobs=64, verbose=1000, backend='multiprocessing')(delayed(compressFootprint)(files) for files in footprintBatches)
    # for idx, files in enumerate(footprintBatches):
    #     p = multiprocessing.Process(target=compressFootprint, args=([files]))
    #     processes.append(p)
    #     p.start()
        
    # for process in tqdm(processes):
    #     process.join()
    
    # print('That took {} seconds'.format(time.time() - starttime))