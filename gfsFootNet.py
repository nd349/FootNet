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
from glob import glob
import matplotlib.pyplot as plt
from shapely import Polygon, Point
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from math import asin, atan2, cos, degrees, radians, sin

import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from unetpp_model import NestedUNet

class metGFSSurface(Dataset):
    def __init__(self, receptors, files, lons, lats, met_DIR, var_list, met_type='GFS', interpolate_method='cubic', met_data={}):
        self.receptors = receptors
        self.files = files
        time_list = list(set([val[2] for val in receptors]))
        time_list = [datetime.datetime.strptime(val, "%Y%m%d%H") for val in time_list]
        print(time_list)
        self.time_list = time_list
        self.lons = lons
        self.lats = lats
        self.met_DIR = met_DIR
        self.met_type = met_type
        self.var_list = var_list # var_list decides surface or column-averaged features
        self.interpolate_method = 'cubic'
        self.met_data = met_data
        self.input_met_dict = {}
        self.get_input_met_dict(time_list, lons, lats)

    def get_input_met_dict(self, time_list, lons, lats):
        # import pdb; pdb.set_trace()
        for tstamp in tqdm(time_list):
            tstamp_list = [tstamp+datetime.timedelta(hours=-hist) for hist in [0, 6, 12, 18, 24]]
            # tstamp = datetime.datetime.strftime(tstamp, "%Y%m%d%H")
            # tstamp_list = [datetime.datetime.strftime(val, "%Y%m%d%H") for val in tstamp_list]
            for dt in tstamp_list:
                if dt not in self.input_met_dict:
                    self.input_met_dict[dt] = self.get_met_data_lite(dt)
        print(self.input_met_dict.keys(), self.met_data.keys())

    def __len__(self):
        return len(self.receptors)

    def __getitem__(self, idx):
        rlon, rlat, tstamp = self.receptors[idx]
        rlon = float(rlon)
        rlat = float(rlat)
        tstamp = datetime.datetime.strptime(tstamp, "%Y%m%d%H")
        tstamp_list = [tstamp+datetime.timedelta(hours=-hist) for hist in [0, 6, 12, 18, 24]]
        # print(tstamp_list)

        # for key in self.input_met_dict:
        #     print(np.max(self.input_met_dict[key][0]), np.max(self.input_met_dict[key][1]), np.max(self.input_met_dict[key][2]), np.max(self.input_met_dict[key][3]))
        
        data = []
        for dt in tstamp_list:
            data.append(self.input_met_dict[dt].copy())
        
        dist = self.get_distance(tstamp, rlon, rlat, self.lats, self.lons)
        uxy_list = [np.mean(data[i][0]) for i in range(len(data))]
        vxy_list = [np.mean(data[i][1]) for i in range(len(data))]
        comb_plume, gp_separate = self.get_gaussian_plume_inputs_single_pixel(uxy_list, vxy_list, rlon, rlat)
        gp0 = self.zstandard(gp_separate[:, :, 0])[np.newaxis, :,:]
        comb_plume = np.array(comb_plume)[np.newaxis, :, :]
        comb_plume[np.where(comb_plume>=0.08)] = 1
        comb_plume[np.where(comb_plume<0.08)] = 0
        # import pdb; pdb.set_trace()
        tempx = self.transform_function24h(data[0], data[1], data[2], data[3], data[4])
        tempx = np.concatenate([gp0, tempx, comb_plume, dist[np.newaxis, :, :], np.exp(0.01*dist)[np.newaxis, :, :]], axis=0)
        return tempx, self.files[idx]

    def zstandard(self, arr):
        _mu = np.nanmean(arr)
        _std = np.nanstd(arr)
        return (arr - _mu)/_std
        
    def transform_function24h(self, _xx, _6xx, _12xx, _18xx, _24xx):
        '''
        xx: (400, 400, 14)
        yy: (400, 400)
        predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
                  'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
        '''
        
        ###
        # typical mean values:
        # [ 3.77901167e-02 -7.10877708e-02  1.24484683e+00  2.56569862e+02
        #   9.80964342e+02  2.82531180e+02  2.88608260e+02  1.18414726e+00
        #   7.51264114e+00  9.86283611e-01  1.14933095e-02  8.46494663e+02
        #   2.86330624e+02  1.02993263e+02]
        #          'U10M', 'V10M', 'PBLH', 'PRSS'
        SCALERS = [1e1,     1e1,    1e-3,   1e-3]
        # SCALERS = [1, 1, 1, 1]
        BIAS =    [  0,     0,       0,       0,      0,     0,     0]
        
        # _xx, _yy = cropx(xx, yy)
        for i in range(len(SCALERS)):
            _xx[i, :, :] = _xx[i, :, :]*SCALERS[i] #+ BIAS[i]
    
        
        # _6xx, _ = cropx(_6xx, yy)
        for i in range(len(SCALERS)):
            _6xx[i, :, :] = _6xx[i, :, :]*SCALERS[i] #+ BIAS[i]
        
        # _12xx, _ = cropx(_12xx, yy)
        for i in range(len(SCALERS)):
            _12xx[i, :, :] = _12xx[i, :, :]*SCALERS[i] #+ BIAS[i]
            
        # _18xx, _ = cropx(_18xx, yy)
        for i in range(len(SCALERS)):
            _18xx[i, :, :] = _18xx[i, :, :]*SCALERS[i]
    
        # _24xx, _ = cropx(_24xx, yy)
        for i in range(len(SCALERS)):
            _24xx[i, :, :] = _24xx[i, :, :]*SCALERS[i]
            
        return np.concatenate([_xx, _6xx, _12xx, _18xx, _24xx], axis=0)

    def interpolate1DLatLon(self, coarse_grid, fine_grid, var_list, tidx, fh, interpolate_method):
        # import pdb; pdb.set_trace()
        if var_list:
            metlon, metlat = coarse_grid
            metlon = (metlon+180)%360-180
            lons, lats = fine_grid
            lat_mask = (metlat>=(lats[0]-1)) & (metlat<=(lats[-1]+1))
            lon_mask = (metlon>=(lons[0]-1)) & (metlon<=(lons[-1]+1))
            lat_roi = metlat[lat_mask]
            lon_roi = metlon[lon_mask]
            metlon_mesh, metlat_mesh = np.meshgrid(metlon, metlat)
            lons_mesh, lats_mesh = np.meshgrid(lons, lats)
            points = np.array([metlat_mesh[lat_mask][:, lon_mask].ravel(), metlon_mesh[lat_mask][:, lon_mask].ravel()]).T
            output = np.zeros((len(var_list), lats.shape[0], lons.shape[0]))
            for i, var in enumerate(var_list):
                temp = np.array(fh[var])[tidx][np.ix_(lat_mask, lon_mask)]
                values = temp.ravel()
                temp_highres = griddata(points, values, (lats_mesh, lons_mesh), method=interpolate_method)
                output[i, :, :] = temp_highres
            return output
        else:
            raise Exception("No variables are provided")

    def get_met_data_lite(self, dt):
        _yy, _mm, _dd, _hh = dt.year, dt. month, dt.day, dt.hour
        metfile = self.get_met_file(_yy, _mm, _dd, _hh)
        if metfile not in self.met_data:
            self.met_data[metfile] = self.open_data(metfile, self.met_type)
            fh = self.met_data[metfile]
        else:
            fh = self.met_data[metfile]
            
        times = [pd.to_datetime(int(val), unit='ns') for val in np.array(fh['time'])]
        tidx = np.argmin(np.abs(np.array(times) - dt))
        metlon, metlat = np.array(fh['x']),np.array(fh['y'])
        coarse_grid = (metlon, metlat)
        fine_grid = (self.lons, self.lats)
        var_list = self.var_list
        interpolate_method = self.interpolate_method
        return self.interpolate1DLatLon(coarse_grid, fine_grid, var_list, tidx, fh, interpolate_method)
        

    def get_met_file(self, yy, mm, dd, hh):
        # 0, 6, 12, 18
        # hhh = [0, 6, 12, 18]
        # hidx = int(hh//6)
        if self.met_type == 'GFS':
            return self.met_DIR + '%04d%02d%02d_gfs0p25'%(yy, mm, dd)
        # return HRRR_DIR + 'hysplit.%04d%02d%02d.%02dz.hrrra'%(yy, mm, dd, hhh[hidx])

    def open_data(self, file, met_type):
        if met_type in ['GFS', 'HRRR']:
            return xr.open_dataset(file, engine='pseudonetcdf')
        else:
            return xr.open_dataset(file)

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

    def get_distance(self, timestamp, rlon, rlat, lat, lon):
        # print(file)
        # print(file.split("/")[-1].split("_")[3:6])
        # timestamp, rlon, rlat = file.split("/")[-1].split("_")[3:6]
        # rlon = float(rlon)
        # rlat = float(rlat)
        # # print(timestamp, rlon, rlat)
        # data = nc.Dataset(file)
        # lat = np.array(data['lat'])
        # lon = np.array(data['lon'])
        
        
        rlat_index = np.unravel_index((np.abs(lat- rlat)).argmin(), lat.shape)
        rlon_index = np.unravel_index((np.abs(lon- rlon)).argmin(), lon.shape)
        # print(rlat_index, rlon_index, rlon, rlat)
        # data.close()
        
        lon, lat = np.meshgrid(lon, lat)
        
        lon = lon*np.pi/180
        lat = lat*np.pi/180
        rlon = rlon*np.pi/180
        rlat = rlat*np.pi/180
    
        a = np.sin((lat-rlat)/2)**2 + np.cos(lat)*np.cos(rlat)*(np.sin((lon-rlon)/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371e3*c/1000
        return d

def write_emulated_footprint(foot, timestamp, receptor_lon, receptor_lat, path):
    file = f"{path}emulator_{timestamp}_{receptor_lon}_{receptor_lat}.nc"
    out_nc = nc.Dataset(file, "w", format="NETCDF4")
    out_nc.createDimension("lat", lats.shape[0])
    out_nc.createDimension("lon", lons.shape[0])
    out_nc.createDimension("info", 1)
    
    lat = out_nc.createVariable("lat", np.float32, ("lat",))
    lon = out_nc.createVariable("lon", np.float32, ("lon",))
    val = out_nc.createVariable("foot", np.float32, ("lat", "lon"))
    rlat = out_nc.createVariable("receptor_lat", np.float32, ("info"))
    rlon = out_nc.createVariable("receptor_lon", np.float32, ("info"))
    clon_y = out_nc.createVariable("clon", np.float32, ("info"))
    clat_y = out_nc.createVariable("clat", np.float32, ("info"))
    lat[:] = lats
    lon[:] = lons
    val[:, :] = foot
    rlat[:] = receptor_lat
    rlon[:] = receptor_lon
    clon_y[:] = clon
    clat_y[:] = clat
    out_nc.close()

# FootNet
def resume(model, optimizer, filename):
    print(f"...Loading {filename}")
    checkpoint = torch.load(filename)
    print("EPOCH number:", checkpoint['EPOCHS_RUN'])
    print("Validation loss:", checkpoint['valid_loss'])
    model.load_state_dict(checkpoint['MODEL_STATE'])

def inference(a):
    # a = -np.sqrt(a)
    epsilon = 1e-3
    a = a/1000
    a = a + np.log(epsilon)
    a = np.exp(a)-epsilon
    a[np.where(a<0)] = 0
    return a

def generate_footprints(test_DG):
    unet.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_DG)):
            inputs, files = data
            inputs = inputs.to(device, dtype=torch.float)
            prediction = unet(inputs)
            
            b = inference(prediction.cpu().detach().numpy())
            c = files
            for i in range(inputs.shape[0]):
                # file = f"{stilt_path}{files[i].split('/')[-1][4:]}"
                timestamp, receptor_lon, receptor_lat = files[i].split('/')[-1].split("_")[:3]
                timestamp = timestamp[:10]
                foot = b[i, 0]
                write_emulated_footprint(foot, timestamp, float(receptor_lon), float(receptor_lat), output_path)

met_DIR = PATH
output_path = PATH
var_list = ["U10M", "V10M", "PBLH", "PRSS"]

num_lats, num_lons = 481, 601
full_xLim = [ -125.0, -120.0 ]
full_yLim = [   36.0,   40.0 ]
orig_lats = np.linspace(full_yLim[0], full_yLim[1], num_lats)
orig_lons = np.linspace(full_xLim[0], full_xLim[1], num_lons)
clon_index = int(orig_lons.shape[0]/2)
clat_index = int(orig_lats.shape[0]/2)
clon = orig_lats[clon_index]
clat = orig_lons[clat_index]
lats = orig_lats[clat_index-200:clat_index+200]
lons = orig_lons[clon_index-200:clon_index+200]

    
experiment = "24hNestedUNet_scaling_factors"
location = PATH
snapshot_path = location + f"best_model{experiment}.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = NestedUNet(input_channels=24, num_classes=1)

unet = unet.to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-4)
resume(unet, optimizer, f"{location}best_model{experiment}.pth")
print(optimizer)

test2020_df = pd.read_csv("../../../beacon/Covid2020_beacon_generalizable.csv")
receptors_df = pd.DataFrame(glob("/home/disk/hermes2/nd349/data/STILT_CONUS/BEACO2N/GFS/footprints/*.nc"), columns=['path'])

test2020_df['check'] = test2020_df['path'].apply(lambda x:"|".join(x.split("/")[-1].split("_")[3:6]))
receptors_df['check'] = receptors_df['path'].apply(lambda x:"|".join(str(val) for val in [x.split("/")[-1].split("_")[0][:10], round(float(x.split("/")[-1].split("_")[1]), 3), round(float(x.split("/")[-1].split("_")[2]), 3)]))
gfs_receptors = receptors_df.merge(test2020_df, how='left', on='check')
gfs_receptors['receptors'] = gfs_receptors['check'].apply(lambda x:[float(x.split("|")[1]), float(x.split("|")[2]), x.split("|")[0]])
gfs_receptors['date'] = gfs_receptors['check'].apply(lambda x:x.split("|")[0][:8])
receptors = list(gfs_receptors['receptors'])
files = list(gfs_receptors['path_x'])
grouped = gfs_receptors.groupby('date')

for idx, group in grouped:
    group = group.reset_index(drop=True)
    receptors = list(group['receptors'])
    files = list(group['path_x'])
    # hrrr_files = [val for val in group['path_y']]
    # hrrr_files = list(group['path_y'])
    # hrrr_DG = DataLoader(FootDataset(hrrr_files, transform=transform, extension='.nc', backhours=backhours),  batch_size=batch_size,
    #                                           shuffle=False, num_workers=8, pin_memory=True)
    met = metGFSSurface(receptors, files, lons, lats, met_DIR, var_list, met_data={})
    gfs_DG = DataLoader(met,  batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    generate_footprints(gfs_DG)
    
    