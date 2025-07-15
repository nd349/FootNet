# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nd349
# @Last Modified time: 2024-07-16 13:34:40

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import glob
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import netCDF4 as nc
import geopandas as gpd
import shapely
from shapely import Polygon, Point
from scipy.spatial import Delaunay

import time
# from config import *

def interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out, d=2):
    xy=np.zeros([grid_x_in.shape[0]*grid_x_in.shape[1],2])
    uv=np.zeros([grid_x_out.shape[0]*grid_x_out.shape[1],2])
    xy[:,0] = grid_x_in.flatten('F')
    xy[:,1] = grid_y_in.flatten('F')
    uv[:,0] = grid_x_out.flatten('F')
    uv[:,1] = grid_y_out.flatten('F')
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def regmet(data, vtx, wts, out_dims):
    '''
    Read and regrid met fields to given longitudes/latitudes
    '''
    data_out = interpolate(data.flatten('F'),vtx,wts).reshape(out_dims, order='F')
    return data_out


def get_hrrr_file(yy, mm, dd, hh, HRRR_DIR):
    # 0, 6, 12, 18
    hhh = [0, 6, 12, 18]
    hidx = int(hh//6)
    return HRRR_DIR + '%04d/hysplit.%04d%02d%02d.%02dz.nc'%(yy, yy, mm, dd, hhh[hidx])
    # return HRRR_DIR + 'hysplit.%04d%02d%02d.%02dz.hrrra'%(yy, mm, dd, hhh[hidx])


def get_met_column_data_lite(footlons, footlats, timestamp, HRRR_DIR, trimsize, hr3lat_full, hr3lon_full, hist=0):
    predlist = ['U10M', 'V10M'] # 2
    reftime = datetime.datetime(1950, 1, 1, 0, 0, 0, 0)
    
    clon = footlons[int(footlons.shape[0]/2)]
    clat = footlats[int(footlats.shape[0]/2)]
    dtnow = datetime.datetime.strptime(timestamp[:10], "%Y%m%d%H")
    histdt = dtnow + datetime.timedelta(hours=hist)
    _yy, _mm, _dd, _hh = histdt.year, histdt. month, histdt.day, histdt.hour
    h3rfile = get_hrrr_file(_yy, _mm, _dd, _hh, HRRR_DIR)
    
    fh = nc.Dataset(h3rfile)
    # fh = xr.open_dataset(h3rfile, engine='pseudonetcdf')
    # fh = fh_dict[histdt]
    h3r_data = fh.variables
    
    
    # times = fh.coords['time'].values
    times = [pd.to_datetime(int(val), unit='ns') for val in np.array(fh['time'])]
    # times = pd.to_datetime(times)
    # tdelt = (times - histdt).seconds
    tidx = np.argmin(np.abs(np.array(times) - histdt))
    # print(h3rfile, tidx, histdt)
    
    distances = (hr3lon_full - clon)**2 + (hr3lat_full - clat)**2
    cind = np.argwhere(distances == np.min(distances))[0]
    cxind = cind[0] # lat
    cyind = cind[1] # lon
    hr3lon = hr3lon_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    hr3lat = hr3lat_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    
    # Regridding weights
    grid_x_in, grid_y_in = hr3lon, hr3lat
    grid_x_out, grid_y_out = np.meshgrid(footlons, footlats)
    vtx, wts = interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out)
    
    _u10m = np.array(h3r_data['U10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
    _u10mr = regmet(_u10m, vtx, wts, grid_x_out.shape)
    
    _v10m = np.array(h3r_data['V10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
    _v10mr = regmet(_v10m, vtx, wts, grid_x_out.shape)
    
    output = np.zeros((footlons.shape[0], footlats.shape[0], len(predlist)))
    output[:, :, 0] = _u10mr
    output[:, :, 1] = _v10mr
    return output


def get_met_data(lons, lats, timestamp, met_dict, HRRR_DIR, trimsize, hr3lat_full, hr3lon_full):
    # global met_dict
    met_timestamps = [timestamp + datetime.timedelta(hours=-i*6) for i in range(0, 13)]
    # print(timestamp, met_timestamps)
    for met_timestamp in met_timestamps:
        if met_timestamp not in met_dict:
            met_dict[met_timestamp] = get_met_column_data_lite(lons, lats, datetime.datetime.strftime(met_timestamp, "%Y%m%d%H"), HRRR_DIR, trimsize, hr3lat_full, hr3lon_full)
    return met_dict

def fill_missing_values(bkg, bkg_error, background_date_range):
    bkg = list(pd.Series(bkg, index=background_date_range).interpolate(method='linear').ffill().bfill())
    bkg_error = pd.Series(bkg_error, index=background_date_range)
    bkg_error[bkg_error==0] = None
    bkg_error = list(bkg_error.interpolate(method='linear').ffill().bfill())
    return bkg, bkg_error

def get_background(date, octant, background):
    background_date_range = background.background_date_range
    index = list(background_date_range).index(date)
    # print(date, octant, index)
    return background.background_dict[octant]['bkg'][index], background.background_dict[octant]['bkg_error'][index]

def get_angle_from_meridional_axis(u, v):
    return (np.degrees(np.arctan2(u, v))+360)%360


def get_avg_wind_resolved(idx, tstamp, avg_foot_run_hours, background, mode='resolved', transport_max_backhours=None):
    # tstamp = H_Y_dict[idx]['time']
    if mode == 'NA':
        # avg_foot_run_hours = int(H_Y_dict[idx]['avg_foot_run_hours'])
        if avg_foot_run_hours%6 > 4:
            range_start_lim = avg_foot_run_hours//6
        else:
            range_start_lim = avg_foot_run_hours//6 - 1
        range_end_lim = min(avg_foot_run_hours//6 + 2, int(transport_max_backhours/6))
        met_timestamps = [tstamp + datetime.timedelta(hours=-i*6) for i in range(range_start_lim, range_end_lim)]
    else:
        met_timestamps = [tstamp + datetime.timedelta(hours=-i*6) for i in range(1, 13)]
    winds = [(np.average(background.met_dict[key][:, :, 0]), np.average(background.met_dict[key][:, :, 1])) for key in met_timestamps]
    # print(winds)

    reversed_wind_direction = [get_angle_from_meridional_axis(-u, -v) for (u, v) in winds]
    return reversed_wind_direction, met_timestamps


def get_background_value(idx, tstamp, avg_foot_run_hours, background, transport_max_backhours):
    wind_directions, met_timestamps = get_avg_wind_resolved(idx, tstamp, avg_foot_run_hours, background, transport_max_backhours=transport_max_backhours)
    bkg_values = []
    bkg_error_values = []
    for jdx, angle in enumerate(wind_directions):
        octant = None
        met_tstamp = met_timestamps[jdx]
        if  0 <= angle <= 45:
            octant = 1
        elif 45 <= angle <= 90:
            octant = 2
        elif 90 <= angle <= 135:
            octant = 3
        elif 135 <= angle <= 180:
            octant = 4
        elif 180 <= angle <= 225:
            octant = 5
        elif 225 <= angle <= 270:
            octant = 6
        elif 270 <= angle <= 315:
            octant = 7
        elif 315 <= angle <= 360:
            octant = 8
            
        met_tstamp_bkg_values = get_background(met_tstamp - datetime.timedelta(hours=met_tstamp.hour), octant, background)
        
        bkg_values.append(met_tstamp_bkg_values[0])
        bkg_error_values.append(met_tstamp_bkg_values[1])
    
    
    bkg = np.average(bkg_values)
    bkg_error = np.average(bkg_error_values)
    # print(idx, H_Y_dict[idx]['time'], bkg, bkg_error)
    return bkg, bkg_error


def get_larger_domain_obs_background(domain_obs):
    print("Loading larger domain obs for background ....")
    domain_df = pd.read_csv(domain_obs)
    domain_df['geometry'] =  domain_df.apply(lambda x: Polygon(zip([x.iloc[13], x.iloc[14], x.iloc[15], x.iloc[16]], [x.iloc[9], x.iloc[10], x.iloc[11], x.iloc[12]])), axis=1)
    domain_gdf = gpd.GeoDataFrame(domain_df, geometry = 'geometry')
    return domain_gdf

class background():
    def __init__(self, dk, domain_gdf, HRRR_DIR, trimsize, lons, lats, hr3lat_full, hr3lon_full, octant_dict, background_date_range):
        self.timestamp_list = list(set(dk['time']))
        self.met_dict = {}
        self.HRRR_DIR = HRRR_DIR
        self.trimsize = trimsize
        self.lats = lats
        self.lons = lons
        self.hr3lat_full = hr3lat_full
        self.hr3lon_full = hr3lon_full
        self.background_date_range = background_date_range

        for tstamp in tqdm(self.timestamp_list):
            self.met_dict = get_met_data(lons, lats, tstamp, self.met_dict, self.HRRR_DIR, self.trimsize, self.hr3lat_full, self.hr3lon_full,)


        self.background_dict = {}
        for octant in octant_dict:
            print("Octant:", octant)
            self.background_dict[octant] = {}
            self.background_dict[octant]['bkg'] = [None for i in range(self.background_date_range.shape[0])]
            self.background_dict[octant]['bkg_error'] = [None for i in range(self.background_date_range.shape[0])]
            self.background_dict[octant]['count'] = []
            for jdx, date in enumerate(self.background_date_range):
                # print(date)
                concs = list(domain_gdf[(domain_gdf['reference_time']==datetime.datetime.strftime(date, "%Y-%m-%d")) & \
                    (octant_dict[octant][0] < domain_gdf['lat_corner0']) & (octant_dict[octant][1] > domain_gdf['lat_corner0']) & \
                    (octant_dict[octant][0] < domain_gdf['lat_corner1']) & (octant_dict[octant][1] > domain_gdf['lat_corner1']) & \
                    (octant_dict[octant][0] < domain_gdf['lat_corner2']) & (octant_dict[octant][1] > domain_gdf['lat_corner2']) & \
                    (octant_dict[octant][0] < domain_gdf['lat_corner3']) & (octant_dict[octant][1] > domain_gdf['lat_corner3']) & \
                    (octant_dict[octant][2] < domain_gdf['lon_corner0']) & (octant_dict[octant][3] > domain_gdf['lon_corner0']) & \
                    (octant_dict[octant][2] < domain_gdf['lon_corner1']) & (octant_dict[octant][3] > domain_gdf['lon_corner1']) & \
                    (octant_dict[octant][2] < domain_gdf['lon_corner2']) & (octant_dict[octant][3] > domain_gdf['lon_corner2']) & \
                    (octant_dict[octant][2] < domain_gdf['lon_corner3']) & \
                    (octant_dict[octant][3] > domain_gdf['lon_corner3'])]['methane_mixing_ratio_bias_corrected'])
                
                self.background_dict[octant]['count'].append(len(concs))
                if concs:
                    self.background_dict[octant]['bkg'][jdx] = np.average(concs)
                    self.background_dict[octant]['bkg_error'][jdx] = np.std(concs)

            self.background_dict[octant]['bkg'], self.background_dict[octant]['bkg_error'] = fill_missing_values(self.background_dict[octant]['bkg'], self.background_dict[octant]['bkg_error'], background_date_range)
        # print("met dict:", self.met_dict.keys())
        # print("background dict:", self.background_dict)


