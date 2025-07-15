# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-05-28 19:16:36

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
from Utils.background import *
from torch.utils.data import Dataset, DataLoader
from FootNet.dataset import FootDataset


class ReadFootprints():
    def __init__(self, transport_max_backhours, nrow, ncol, transport_max_days,clon_index=None, clat_index=None):
        self.transport_max_backhours = transport_max_backhours
        self.nrow = nrow
        self.ncol = ncol
        self.transport_max_days = transport_max_days
        self.clon_index = clon_index
        self.clat_index = clat_index
        pass

    def get_GP_footprint(self, data, model, mode):
        if model == "XSTILT":
            if mode == "resolved":
                # foot_paths = list(data['foot_path'])
                # gp_foot = self.read_XSTILT_GP_footprint(foot_paths)
                gp_foot = self.compute_daily_avg_GP_footprint(data)
                return gp_foot

    def compute_daily_avg_GP_footprint(self, data):
        foot_paths = list(data['foot_path'])
        temp = np.zeros((len(foot_paths), self.transport_max_backhours, self.nrow, self.ncol)) 
        avg_foot_run_hours = 0
        for jdx, path in enumerate(foot_paths):
            subpixel_data = nc.Dataset(path)
            subpixel_foot = np.array(subpixel_data['foot'])
            if self.clat_index and self.clon_index:
                subpixel_foot = subpixel_foot[:, self.clat_index-200:self.clat_index+200, self.clon_index-200:self.clon_index+200]
            avg_foot_run_hours += subpixel_foot.shape[0]
            subpixel_data.close()
        
            for kdx in range(1, subpixel_foot.shape[0]+1):
                # print(kdx)
                temp[jdx, -kdx, :, :] = subpixel_foot[-kdx, :, :]
        
            gp_foot = np.average(temp, axis=0)
            
            st_end = data['time'][0] + datetime.timedelta(hours=23-data['time'][0].hour)
            et_end = st_end - datetime.timedelta(hours=self.transport_max_days*24 - 1)
            foot_date_range = pd.date_range(start=et_end, end=st_end, freq='1h')
            foot_tstamp_indices = [idx for idx in range(foot_date_range.get_loc(data['time'][0])-(self.transport_max_backhours-1), foot_date_range.get_loc(data['time'][0])+1)]
            foot_H = np.zeros((foot_date_range.shape[0], self.nrow, self.ncol))
            for idx, index in enumerate(foot_tstamp_indices):
                foot_H[index, :, :] = gp_foot[idx, :, :]
        
            h3 = np.average(foot_H[-24:], axis=0)
            h2 = np.average(foot_H[-48:-24], axis=0)
            h1 = np.average(foot_H[-72:-48], axis=0)
            h0 = np.average(foot_H[-96:-72], axis=0)
            h_total = np.zeros((self.transport_max_days, self.nrow, self.ncol))
            h_total[0, :, :] = h0
            h_total[1, :, :] = h1
            h_total[2, :, :] = h2
            h_total[3, :, :] = h3
            
        avg_foot_run_hours = avg_foot_run_hours/data.shape[0]
            
        return h_total, foot_date_range, avg_foot_run_hours

    # def read_XSTILT_GP_footprint(self, foot_paths, nrow, ncol, clat_index=None, clon_index=None):
    #     temp = np.zeros((len(foot_paths), 3, nrow, ncol))
    #     for jdx, path in enumerate(foot_paths):
    #         subpixel_data = nc.Dataset(path)
    #         subpixel_foot = np.array(subpixel_data['foot'])
    #         if clat_index and clon_index:
    #             subpixel_foot = subpixel_foot[:, clat_index-200:clat_index+200, clon_index-200:clon_index+200]
    #         subpixel_data.close()
    #         foot_24h = np.nansum(subpixel_foot[-24:], axis=0)
    #         foot_48h = np.nansum(subpixel_foot[-48:-24], axis=0)
    #         foot_72h = np.nansum(subpixel_foot[-72:-48], axis=0)
    #         temp[jdx, 0, :, :] = foot_72h
    #         temp[jdx, 1, :, :] = foot_48h
    #         temp[jdx, 2, :, :] = foot_24h
    
    
    #     gp_foot = np.zeros((3, 400, 400))
    #     if np.max(foot_72h) > 0:
    #         # print("72")
    #         gp_foot[0, :, :] = np.nanmean(temp[:, 0, :, :], axis=0)
    #     if np.max(foot_48h) > 0:
    #         # print("48")
    #         gp_foot[1, :, :] = np.nanmean(temp[:, 1, :, :], axis=0)
    #     if np.max(foot_24h) > 0:
    #         # print("24")
    #         gp_foot[2, :, :] = np.nanmean(temp[:, 2, :, :], axis=0) 
    #     return gp_foot #, temp


def get_Obs_Foot_Data(idx, data, readFoot, foot_model, mode='', footnet_model=None, input_met=None, batch_size=None, num_workers=None):
    # print(idx)
    # global met_dict
    data = data.reset_index(drop=True)
    group_dict = {}
    group_dict['methane_mixing_ratio_bias_corrected'] = list(data['methane_mixing_ratio_bias_corrected'])[0]
    group_dict['methane_mixing_ratio'] = list(data['methane_mixing_ratio'])[0]
    group_dict['methane_mixing_ratio_precision'] = list(data['methane_mixing_ratio_precision'])[0]
    group_dict['time'] = list(data['time'])[0]
    group_dict['ground_pixel'] = data['polygon_ground_pixel_geometry'][0]
    group_dict['rlat'] = data['lati']
    group_dict['rlon'] = data['long']
    
    if foot_model == "XSTILT":
        group_dict['foot_path'] = list(data['foot_path'])
         # H_Y_dict[idx]['subpixel_foot'] = np.zeros((data.shape[0], 3, 400, 400))
        # group_dict['gp_foot'], group_dict['subpixel_foot'] = read_subpixel_footprints(group_dict['foot_path'])
        group_dict['gp_foot'], group_dict['foot_timestamps'], group_dict['avg_foot_run_hours'] = readFoot.get_GP_footprint(data, \
            model=foot_model, mode=mode)

    elif foot_model == "FootNet":
        receptors = []
        for index in range(data.shape[0]):
            receptors.append([data['time'][index], data['long'][index], data['lati'][index], index])

        # print("Forming dataloader")
        batch_DG = DataLoader(FootDataset(receptors, input_met),  batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        # import pdb; pdb.set_trace()
        # print(f"Running inference on {len(batch_DG)} batches")
        foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, foot_hours = footnet_model.run_inference(batch_DG)
        group_dict['gp_foot'], group_dict['foot_timestamps'], group_dict['avg_foot_run_hours'] = \
        footnet_model.compute_daily_resolved_footprint(foots, foot_hours, data, readFoot.nrow, readFoot.ncol, readFoot.transport_max_days, \
            readFoot.transport_max_backhours)
        group_dict['footnet_reference'] = [reference_indices, reference_timestamps, reference_rlons, reference_rlats]
    return group_dict, idx

def get_obs(obs_file):
    print("Loading focused area domain obs ...")
    df = pd.read_csv(obs_file)
    df['time'] = df['time'].apply(lambda x:datetime.datetime.strptime(x, "%Y/%m/%d %H"))
    df['polygon_ground_pixel_geometry'] = df['ground_pixel_geometry'].apply(lambda x:shapely.wkt.loads(x))
    df['geometry'] = df['geometry'].apply(lambda x:shapely.wkt.loads(x))
    df = df.dropna().reset_index(drop=True)

    count = 0
    dk = df.copy()
    for val in list(df[df['count_found_foots']!=1]['ground_pixel_geometry']):
        count += dk[dk['ground_pixel_geometry']== val].shape[0]
        dk =  dk[dk['ground_pixel_geometry']!= val].reset_index(drop=True)
    print("Bad obs count:", count)
    return df

def get_obs_time_domain(df, m_start, m_end):
    dk = df[(df['time']>=m_start)&(df['time']<=m_end)].reset_index(drop=True)
    return dk

def get_H_Y_dict(dk, readFoot, foot_model, background, mode, footnet_model=None, input_met=None, batch_size=None, num_workers=None, transport_max_backhours=None):
    # dk = dk[dk.index<100].reset_index(drop=True)
    grouped = dk.groupby(['ground_pixel_geometry', 'time'])

    # for idx, data in grouped:
    #     break
    # OUTPUT = Parallel(n_jobs=64, verbose=1000, backend="multiprocessing")(delayed(get_Obs_Foot_Data)(idx, data, readFoot, foot_model=foot_model) for idx, data in [(idx, data)])
    if foot_model == "XSTILT":
        OUTPUT = Parallel(n_jobs=num_workers, verbose=1000, backend="multiprocessing")(delayed(get_Obs_Foot_Data)(idx, data, readFoot, \
            foot_model=foot_model, mode=mode) for idx, data in grouped)
    elif foot_model == "FootNet":
        OUTPUT = []
        for idx, data in tqdm(grouped):
            OUTPUT.append(get_Obs_Foot_Data(idx, data, readFoot, foot_model=foot_model, mode=mode, footnet_model=footnet_model, input_met=input_met, batch_size=batch_size, num_workers=num_workers))

    H_Y_dict = {}
    # import pdb; pdb.set_trace()
    for idx, val in tqdm(enumerate(OUTPUT)):
        # print(idx)
        H_Y_dict[val[1]] = val[0]
        tstamp = H_Y_dict[val[1]]['time']
        avg_foot_run_hours = int(H_Y_dict[val[1]]['avg_foot_run_hours'])
        temp = get_background_value(idx, tstamp, avg_foot_run_hours, background, transport_max_backhours=transport_max_backhours)
        H_Y_dict[val[1]]['bkg'] = temp[0]
        H_Y_dict[val[1]]['bkg_error'] = temp[1]


    # print(H_Y_dict)
    return H_Y_dict

