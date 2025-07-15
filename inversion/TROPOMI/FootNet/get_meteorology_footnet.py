import warnings
warnings.filterwarnings("ignore")

import torch, time, datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random, os
import netCDF4 as nc
import xarray as xr
import geopandas as gpd
import shapely
from shapely import Polygon, Point
from scipy.spatial import Delaunay
from math import asin, atan2, cos, degrees, radians, sin


class input_meteorology():
    def __init__(self, time_list, lons, lats, trimsize, hr3lat_full, hr3lon_full, HRRR_DIR, footnet_hours_mode, maximum_domain_trajectory):
        self.time_list = time_list
        self.lons = lons
        self.lats = lats
        self.trimsize = trimsize
        self.hr3lat_full = hr3lat_full
        self.hr3lon_full = hr3lon_full
        self.HRRR_DIR = HRRR_DIR
        self.footnet_hours_mode = footnet_hours_mode
        self.maximum_domain_trajectory = maximum_domain_trajectory
        self.input_met_dict = self.get_input_met_dict(time_list, lons, lats, trimsize)

    def get_gaussian_plume_inputs_single_pixel(self, uxy_list, vxy_list, x_rlon, x_rlat):
        grid_x_out, grid_y_out = np.meshgrid(self.lons, self.lats)
        x = x_rlon
        y = x_rlat
        xlist = []
        ylist = []
        
        gp_dict = {}
        for idx in range(len(uxy_list)):
            uxy = uxy_list[idx]
            vxy = vxy_list[idx]
            xlist.append(x)
            ylist.append(y)
        
            gprs = self.GaussianPlume(grid_x_out,grid_y_out,x, y, -uxy, -vxy)
            gp_dict[idx] = gprs
            
            if -uxy > 0: # longitude increases
                # print("longitude increases")
                bearing = 90
                distance = abs(uxy*6*3600/1000)
                x = self.get_point_at_distance(y, x, distance, bearing=bearing)[1]
            elif -uxy < 0: # longitude decreases
                # print("longitude decreases")
                bearing = 270
                distance = abs(uxy*6*3600/1000)
                x = self.get_point_at_distance(y, x, distance, bearing=bearing)[1]
            
            if -vxy >0: # latitude increases
                # print("latitude increases")
                bearing = 0
                distance = abs(vxy*6*3600/1000)
                y = self.get_point_at_distance(y, x, distance, bearing=bearing)[0]
            elif -vxy < 0: # latitude decreases
                # print("latitude decreases")
                bearing = 180
                distance = abs(vxy*6*3600/1000)
                y = self.get_point_at_distance(y, x, distance, bearing=bearing)[0]
        gp = np.zeros((self.lons.shape[0], self.lats.shape[0]))
        gp_separate = np.zeros((self.lons.shape[0], self.lats.shape[0], 5))
        for key in gp_dict:
            gp += gp_dict[key]
            gp_separate[:, :, key] = gp_dict[key]
        return gp, gp_separate

    def get_input_emulator_single_pixel(self, tstamp, x_rlon, x_rlat):
        tstamp_list = [tstamp+datetime.timedelta(hours=-hist) for hist in [0, 6, 12, 18, 24]]
        tstamp_list = [datetime.datetime.strftime(val, "%Y%m%d%H") for val in tstamp_list]
        _xpreds = self.input_met_dict[tstamp_list[0]]
        _x6hpreds = self.input_met_dict[tstamp_list[1]]
        _x12hpreds = self.input_met_dict[tstamp_list[2]]
        _x18hpreds = self.input_met_dict[tstamp_list[3]]
        _x24hpreds = self.input_met_dict[tstamp_list[4]]
        
        uxy_list = [np.average(v[:, :, 0]) for v in [_xpreds, _x6hpreds, _x12hpreds, _x18hpreds, _x24hpreds]]
        vxy_list = [np.average(v[:, :, 1]) for v in [_xpreds, _x6hpreds, _x12hpreds, _x18hpreds, _x24hpreds]]
        combined_gaussian_plume, gp_separate = self.get_gaussian_plume_inputs_single_pixel(uxy_list, vxy_list, x_rlon, x_rlat)
        foot_hours = self.estimate_foot_hours(uxy_list, vxy_list, self.maximum_domain_trajectory, footnet_hours_mode=self.footnet_hours_mode)
        return _xpreds, _x6hpreds, _x12hpreds, _x18hpreds, _x24hpreds, combined_gaussian_plume, gp_separate, foot_hours
    
    def get_input_met_dict(self, time_list, lons, lats, trimsize, hist=0):
        input_met_dict = {}
        for tstamp in tqdm(time_list):
            tstamp_list = [tstamp+datetime.timedelta(hours=-hist) for hist in [0, 6, 12, 18, 24]]
            # tstamp = datetime.datetime.strftime(tstamp, "%Y%m%d%H")
            tstamp_list = [datetime.datetime.strftime(val, "%Y%m%d%H") for val in tstamp_list]
            for dt in tstamp_list:
                if dt not in input_met_dict:
                    input_met_dict[dt] = self.get_met_column_data_lite(lons, lats, dt, hist=hist, trimsize=trimsize)
        return input_met_dict
    
    def interp_weights(self, grid_x_in, grid_y_in, grid_x_out, grid_y_out, d=2):
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
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
    def regmet(self, data, vtx, wts, out_dims):
        '''
        Read and regrid met fields to given longitudes/latitudes
        '''
        data_out = self.interpolate(data.flatten('F'),vtx,wts).reshape(out_dims, order='F')
        return data_out

    def GaussianPlume(self, lon, lat, fLon, fLat, uu, vv, aA=104, aB=213, wA=6, wB=2):
        '''
        Function to generate a Gaussian plume using wind fields.
        '''
        # Grid info
        nX,nY = lon.shape[1], lon.shape[0]
        c     = np.zeros([nY,nX],dtype=float)
        # Windspeed and direction
        wspd = np.sqrt(uu**2.+vv**2.)   # 2D
        wdir = np.arctan2(vv, uu)
        # Parameters and stability class
        x0     = 1e3
        aA, wA = 104., 6.
        aB, wB = 213., 2.
        a      = (wspd - wA)/(wB - wA)*(aB - aA) + aA
        if a < aA:
            a = aA
        if a > aB:
            a = aB
        # Flatten the matrices
        # lon,lat = np.meshgrid(lon,lat)
        out_dim = c.shape
        xx      = (lon - fLon)/120.*1e3
        yy      = (lat - fLat)/120.*1e3
        r       = np.sqrt(xx**2.+yy**2.)
        phi     = np.arctan2(yy,xx)-wdir
        lx      = r*np.cos(phi)
        ly      = r*np.sin(phi)
        sig     = a*(lx/x0)**0.894
        # import pdb; pdb.set_trace()
        c = 1./(sig*wspd) * np.exp(-0.5 * (ly/sig)**2. )
        
        c = np.ma.masked_array(c)
        c = c.filled(fill_value=0)
        
        c[np.where(np.isnan(c))] = 0.
        # print(c)
        return c

    def get_point_at_distance(self, lat1, lon1, d, bearing, R=6371):
        """
        lat: initial latitude, in degrees
        lon: initial longitude, in degrees
        d: target distance from initial
        bearing: (true) heading in degrees
        R: optional radius of sphere, defaults to mean radius of earth
    
        Returns new lat/lon coordinate {d}km from initial, in degrees
        """
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        a = radians(bearing)
        lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
        lon2 = lon1 + atan2(
            sin(a) * sin(d/R) * cos(lat1),
            cos(d/R) - sin(lat1) * sin(lat2)
        )
        return (degrees(lat2), degrees(lon2),)

    def get_len(self, coords_1, coords_2):
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
        # 0, 6, 12, 18
        hhh = [0, 6, 12, 18]
        hidx = int(hh//6)
        return self.HRRR_DIR + '%04d/hysplit.%04d%02d%02d.%02dz.nc'%(yy, yy, mm, dd, hhh[hidx])
        # return HRRR_DIR + 'hysplit.%04d%02d%02d.%02dz.hrrra'%(yy, mm, dd, hhh[hidx])


    def get_met_column_data_lite(self, footlons, footlats, timestamp, hist=0, trimsize=150):
        # predlist = ['U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'U500', 'V850', 'V500', 'P850', 'T850'] # 10
        predlist = ['U10M', 'V10M', 'PBLH', 'PRSS', 'U850', 'U500', 'V850', 'V500', 'T850'] # 9
        trimsize = self.trimsize
        # (footlons, footlats, timestamp) = receptor
        clon = footlons[int(footlons.shape[0]/2)]
        clat = footlats[int(footlats.shape[0]/2)]
        dtnow = datetime.datetime.strptime(timestamp[:10], "%Y%m%d%H")
        histdt = dtnow + datetime.timedelta(hours=hist)
        _yy, _mm, _dd, _hh = histdt.year, histdt. month, histdt.day, histdt.hour
        h3rfile = self.get_hrrr_file(_yy, _mm, _dd, _hh)
        
        fh = nc.Dataset(h3rfile)
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
        
        # _p850 = np.array(h3r_data['PRES9_850hPa'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
        # _p850r = self.regmet(_p850, vtx, wts, grid_x_out.shape)
        
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
        # output[:, :, 8] = _p850r
        output[:, :, 8] = _t850r
        return output

    def estimate_foot_hours(self, uxy_list, vxy_list, maximum_domain_trajectory, footnet_hours_mode='average'):
        if footnet_hours_mode == 'average':
            u = np.average(uxy_list)
            v = np.average(vxy_list)
            dist_per_hour = np.sqrt((u*3600/1000)**2 + (v*3600/1000)**2) # km
            foot_hours = maximum_domain_trajectory/dist_per_hour # hours
            foot_hours = round(foot_hours)
        elif footnet_hours_mode == "6hourly":
            pass
        return foot_hours