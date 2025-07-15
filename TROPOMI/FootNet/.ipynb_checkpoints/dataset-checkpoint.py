import warnings
warnings.filterwarnings("ignore")

import torch, time, datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class FootDataset(Dataset):
    def __init__(self, data, input_met):
        self.data = data
        self.lons = input_met.lons
        self.lats = input_met.lats
        self.input_met = input_met

    def __len__(self):
        return len(self.data)
    
    def get_distance(self, rlon, rlat, lat, lon):
        rlat_index = np.unravel_index((np.abs(lat- rlat)).argmin(), lat.shape)
        rlon_index = np.unravel_index((np.abs(lon- rlon)).argmin(), lon.shape)
        lon, lat = np.meshgrid(lon, lat)
        lon = lon*np.pi/180
        lat = lat*np.pi/180
        rlon = rlon*np.pi/180
        rlat = rlat*np.pi/180
        a = np.sin((lat-rlat)/2)**2 + np.cos(lat)*np.cos(rlat)*(np.sin((lon-rlon)/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371e3*c/1000
        return d
    
    def zstandard(self, arr):
        _mu = np.nanmean(arr)
        _std = np.nanstd(arr)
        return (arr - _mu)/_std

    def transform_func_12h(self, _xx, _6xx, _12xx, comb_plume, gp_first):
        '''
        xx: (400, 400, 14)
        yy: (400, 400)
        predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
                  'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
        '''
        
        #          'U10M', 'V10M', 'PBLH', 'PRSS', 'UWND9_850hPa', 'VWND9_850hPa', 'UWND17_500hPa', 'VWND17_500hPa', 'PRES9_850hPa', 'TEMP9_850hPa'
        # SCALERS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        SCALERS = [1e1, 1e1, 1e-3, 1e-3, 1, 1, 1, 1, 1e-3, 1e-2]
        BIAS =    [  0,     0,       0,       0,      0,     0,     0, 0, 0, 0]
        
        for i in range(_xx.shape[2]):
            # _xx[:, :, i] = zstandard(_xx[:, :, i])
            _xx[:, :, i] = _xx[:, :, i]*SCALERS[i] #+ BIAS[i]
        
        for i in range(_6xx.shape[2]):
            # _6xx[:, :, i] = zstandard(_6xx[:, :, i])
            _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i] #+ BIAS[i]
        
        
        for i in range(_12xx.shape[2]):
            # _12xx[:, :, i] = zstandard(_12xx[:, :, i])
            _12xx[:, :, i] = _12xx[:, :, i]*SCALERS[i] #+ BIAS[i]
        
        
        comb_plume = np.array(comb_plume)[:, :, np.newaxis]
        comb_plume[np.where(comb_plume>=0.08)] = 1
        comb_plume[np.where(comb_plume<0.08)] = 0
    
        gp_first = self.zstandard(gp_first)
        gp_first = gp_first[:, :, np.newaxis]
        return np.concatenate([gp_first, _xx, _6xx, _12xx, comb_plume], axis=-1)

    def __getitem__(self, idx):
        # print("Index:", idx)
        timestamp, rlon, rlat, index = self.data[idx]
        dist = self.get_distance(rlon, rlat, self.lats, self.lons)
        _pred, _6hpred, _12hpred, _18hpred, _24hpred, combined_gp, gp_separate, foot_hours = self.input_met.get_input_emulator_single_pixel(timestamp, rlon, rlat)

        gp_first = gp_separate[:, :, 0]
        
        # tempx = self.transform_func_12h(_pred, _6hpred, _12hpred, combined_gp, gp_first)
        tempx = self.transform_func_12h(_pred.copy(), _6hpred.copy(), _12hpred.copy(), combined_gp.copy(), gp_first.copy())

        tempxx = np.zeros((tempx.shape[2], tempx.shape[0], tempx.shape[1]))
        
        tempxx = np.concatenate([tempxx, dist[np.newaxis, :, :], np.exp(0.01*dist)[np.newaxis, :, :]], axis=0)
        
        for idx in range(tempx.shape[2]):
            tempxx[idx, :, :] = tempx[:, :, idx]
            
        return tempxx, [index, datetime.datetime.strftime(timestamp, "%Y%m%d%H"), rlon, rlat], foot_hours