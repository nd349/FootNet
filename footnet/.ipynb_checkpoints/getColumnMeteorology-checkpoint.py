import pandas as pd
import numpy as np
import random, os
import netCDF4 as nc
import xarray as xr
import geopandas as gpd
import shapely
import datetime
from scipy.spatial import Delaunay
from tqdm import tqdm

class ColumnMeteorology():
    """
    Class to get column meteorology data from HRRR lite files for given time and locations.
    9 variables: 'U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'V850', 'U500', 'V500', 'T850'
    """
    def __init__(self, time_list, lons, lats, trimsize, hr3lat_full, hr3lon_full, HRRR_DIR, backhours=[0, 6, 12, 18, 24]):
        """
        Init function to set up parameters.
        time_list: list of datetime objects
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        trimsize: int, size to trim the HRRR grid for efficient interpolation
        hr3lat_full: 2D array of full HRRR latitudes (mapping of HRRR grid to y coordinates)
        hr3lon_full: 2D array of full HRRR longitudes (mapping of HRRR grid to x coordinates)
        HRRR_DIR: str, directory path where HRRR lite files are stored
        backhours: list of int, hours to look back for meteorology data
        """
        self.time_list = time_list
        self.lons = lons
        self.lats = lats
        self.trimsize = trimsize
        self.hr3lat_full = hr3lat_full
        self.hr3lon_full = hr3lon_full
        self.HRRR_DIR = HRRR_DIR
        self.backhours = backhours
        self.input_met_dict, self.processed_met_dict = self.get_input_met_dict(time_list, lons, lats, trimsize)

    def get_input_met_dict(self, time_list, lons, lats, trimsize, hist=0):
        """
        Get input meteorology data dictionary for given time list and locations.
        time_list: list of datetime objects
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        trimsize: int, size to trim the HRRR grid for efficient interpolation
        hist: int, hours to look back for meteorology data from the time of measurement.
        Returns two dictionaries: raw input data and processed data.
        1. input_met_dict: {timestamp_str: met_data_array}
        2. processed_input_dict: {timestamp_str: processed_met_data_array}
        """
        input_met_dict = {}
        processed_input_dict = {}
        for tstamp in tqdm(time_list):
            tstamp_list = [tstamp+datetime.timedelta(hours=-hist) for hist in self.backhours]
            # tstamp = datetime.datetime.strftime(tstamp, "%Y%m%d%H")
            tstamp_list = [datetime.datetime.strftime(val, "%Y%m%d%H") for val in tstamp_list]
            for dt in tstamp_list:
                if dt not in input_met_dict:
                    input_met_dict[dt] = self.get_met_column_data_lite(lons, lats, dt, hist=hist, trimsize=trimsize)
                    processed_input_dict[dt] = self.transform_func_24h(input_met_dict[dt].copy())
        return input_met_dict, processed_input_dict

    def transform_func_24h(self, _xx):
        '''
        Transform and scale the input meteorology data
        xx: (400, 400, 9)
        predlist: 'U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'V850', 'U500', 'V500', 'T850'
        '''
        
        #          'U10M','V10M','PBLH','PRSS', 'U850', 'V850', 'U500', 'V500', 'T850'
        # SCALERS = [1, 1, 1, 1]
        SCALERS = [1e1, 1e1, 1e-3, 1e-3, 1, 1, 1, 1, 1e-2]
        BIAS =    [0, 0, 0, 0]
        
        for i in range(_xx.shape[2]):
            # _xx[:, :, i] = zstandard(_xx[:, :, i])
            _xx[:, :, i] = _xx[:, :, i]*SCALERS[i] #+ BIAS[i]
        
        return _xx

    def interp_weights(self, grid_x_in, grid_y_in, grid_x_out, grid_y_out, d=2):
        """
        Calculate interpolation weights for regridding
        from input grid (grid_x_in, grid_y_in) to output grid (grid_x_out, grid_y_out)
        using Delaunay triangulation and barycentric coordinates.
        """
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
    
    def interpolate(self, values, vtx, wts):
        """
        Interpolate values at given vertices with weights
        """
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
    def regmet(self, data, vtx, wts, out_dims):
        '''
        Read and regrid met fields to given longitudes/latitudes
        '''
        data_out = self.interpolate(data.flatten('F'),vtx,wts).reshape(out_dims, order='F')
        return data_out

    def get_len(self, coords_1, coords_2):
        """
        Calculate the great-circle distance between two points on the Earth specified in decimal degrees.
        coords_1: (lat1, lon1)
        coords_2: (lat2, lon2)
        Returns distance in kilometers.
        """
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

    def get_hrrr_file(self, yy, mm, dd, hh):
        """
        Get the HRRR lite file path for given year, month, day, hour.
        """
        # 0, 6, 12, 18
        hhh = [0, 6, 12, 18]
        hidx = int(hh//6)
        return self.HRRR_DIR + '%04d/hysplit.%04d%02d%02d.%02dz.nc'%(yy, yy, mm, dd, hhh[hidx])
        # return HRRR_DIR + 'hysplit.%04d%02d%02d.%02dz.hrrra'%(yy, mm, dd, hhh[hidx]) # For direct loading data from original HRRR files


    def get_met_column_data_lite(self, footlons, footlats, timestamp, hist=0, trimsize=150):
        """
        Get meteorology column data from HRRR lite files for given longitudes, latitudes, and timestamp.
        footlons: 1D array of longitudes
        footlats: 1D array of latitudes
        timestamp: str, in the format "YYYYMMDDHH"
        hist: int, hours to look back for meteorology data from the time of measurement.
        trimsize: int, size to trim the HRRR grid for efficient interpolation.
        Returns a 3D numpy array of shape (len(footlons), len(footlats), 9) with meteorology variables:
        'U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'V850', 'U500', 'V500', 'T850'
        9 variables in total.
        """
        predlist = ['U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'V850', 'U500', 'V500', 'T850'] # 9
        trimsize = self.trimsize
        # (footlons, footlats, timestamp) = receptor
        clon = footlons[int(footlons.shape[0]/2)]
        clat = footlats[int(footlats.shape[0]/2)]
        dtnow = datetime.datetime.strptime(timestamp[:10], "%Y%m%d%H")
        histdt = dtnow + datetime.timedelta(hours=hist)
        _yy, _mm, _dd, _hh = histdt.year, histdt. month, histdt.day, histdt.hour
        h3rfile = self.get_hrrr_file(_yy, _mm, _dd, _hh)
        # print(h3rfile)
        fh = nc.Dataset(h3rfile)
        # fh = xr.open_dataset(h3rfile, engine="pseudonetcdf") # For direct loading data from original HRRR files (very slow)
        h3r_data = fh.variables
        times = [pd.to_datetime(int(val), unit='ns') for val in np.array(fh['time'])]
        tidx = np.argmin(np.abs(np.array(times) - histdt))
        
        distances = (self.hr3lon_full - clon)**2 + (self.hr3lat_full - clat)**2
        cind = np.argwhere(distances == np.min(distances))[0]
        cxind = cind[0] # lat
        cyind = cind[1] # lon
        hr3lon = self.hr3lon_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
        hr3lat = self.hr3lat_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
        
        # Regridding weights
        grid_x_in, grid_y_in = hr3lon, hr3lat
        grid_x_out, grid_y_out = np.meshgrid(footlons, footlats)
        vtx, wts = self.interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out)
        
        _u10m = np.array(h3r_data['U10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _u10mr = self.regmet(_u10m, vtx, wts, grid_x_out.shape)
        
        _v10m = np.array(h3r_data['V10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _v10mr = self.regmet(_v10m, vtx, wts, grid_x_out.shape)
        
        _pblh = np.array(h3r_data['PBLH'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _pblhr = self.regmet(_pblh, vtx, wts, grid_x_out.shape)  
        
        _psfc = np.array(h3r_data['PRSS'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _psfcr = self.regmet(_psfc, vtx, wts, grid_x_out.shape)

        _u850 = np.array(h3r_data['UWND9_850hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _u850r = self.regmet(_u850, vtx, wts, grid_x_out.shape)
        
        _v850 = np.array(h3r_data['VWND9_850hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _v850r = self.regmet(_v850, vtx, wts, grid_x_out.shape)
        
        _u500 = np.array(h3r_data['UWND17_500hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _u500r = self.regmet(_u500, vtx, wts, grid_x_out.shape)
        
        _v500 = np.array(h3r_data['VWND17_500hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _v500r = self.regmet(_v500, vtx, wts, grid_x_out.shape)
        
        _t850 = np.array(h3r_data['TEMP9_850hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        _t850r = self.regmet(_t850, vtx, wts, grid_x_out.shape)
        
        output = np.zeros((footlons.shape[0], footlats.shape[0], len(predlist)))
        output[:, :, 0] = _u10mr
        output[:, :, 1] = _v10mr
        output[:, :, 2] = _pblhr
        output[:, :, 3] = _psfcr
        output[:, :, 4] = _u850r
        output[:, :, 5] = _v850r
        output[:, :, 6] = _u500r
        output[:, :, 7] = _v500r
        output[:, :, 8] = _t850r
        return output