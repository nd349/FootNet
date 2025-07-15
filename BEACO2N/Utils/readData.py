# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-07-22 11:59:53
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 14:41:51

import netCDF4 as nc
import numpy as np
import datetime
from os import listdir
from os.path import isfile, join

def get_files(directory, extension=""):
    """
        Get the relevant files from a given path

        Arguments:
            directory: <str>
            extension: <str>
        returns:
            files: <list>
    """
    if extension:
        files = [f for f in listdir(directory) if f[-len(extension):] == extension]
    else:
        files = [f for f in listdir(directory)]
    files = [directory+file for file in files]
    return files

def read_obs_Data(file, key):
	'''
	Function to read netcdf obs file and return the given key data

	Input
	file: str
	key: str

	Output: numpy array or datetime object
	'''
	try:
		data = nc.Dataset(file)
		if key=='obs_time':
			year = np.array(data['yr'])[0]
			month = np.array(data['mon'])[0]
			day = np.array(data['day'])[0]
			hour = np.array(data['hr'])[0]
			data.close()
			return datetime.datetime(year, month, day, hour)
		else:
			value = np.array(data[key])
			data.close()
			return value
	except Exception as e:
		print(e)


def read_background_conc(file, key):
	'''
	Function to read netcdf background conc file and return the given key data

	Input
	file: str
	key: str

	Output: numpy array
	'''
	try:
		data = nc.Dataset(file)
		value = np.array(data[key])
		data.close()
		return value
		
	except Exception as e:
		print(e)

def read_footprints(foot_info, key, sparse=False):
	'''
	Function to read netcdf footprint file and return the given key data

	Input
	foot_info: tuple
	key: str

	Output: numpy array
	'''
	try:
		if key == 'foot':
			(file_name, lon_ind, lat_ind) = foot_info
			data = nc.Dataset(file)
			foot = np.array(data['foot'])
			data.close()
			return foot
		pass
	except Exception as e:
		print(e)


def read_emissions_data(file, key):
	'''
	Function to read netcdf emission file and return the given key data

	Input
	file: str
	key: str

	Output: value object from netcdf file
	'''
	try:
		data = nc.Dataset(file)
		if key == 'ems_time':
			year = np.array(data['yr'])[0]
			month = np.array(data['mon'])[0]
			day = np.array(data['day'])[0]
			hour = np.array(data['hr'])[0]
			dat.close()
			return datetime.datetime(year, month, day, hour)
		else:
			value = np.array(data[key])
			data.close()
			return value

	except Exception as e:
		print(e)



if __name__ == '__main__':
	file = "/home/disk/hermes/data/footprints/BEACO2N/obs/obs_2018070617_-122.205_38.121_9.318.nc"
	print(read_obs_Data(file, 'obs_time'))