# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2024-06-17 21:31:00

import numpy as np
from tqdm.auto import tqdm

def get_len(coords_1, coords_2):
    """
        Compute the length between two coordinates

        Arguments:
            coords_1: <list>
            coords_2: <list>

        returns:
            <float>
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


def compute_obs_covariance(So_d, tau_time, tau_space, H_Y_dict, indices):
    print("Forming observation error covariance matrix")
    if So_d.shape[0] != len(indices):
        raise Exception(f"So_d shape ({So_d.shape}) and indices shape {len(indices)} are not equal")

    So = np.zeros((So_d.shape[0], So_d.shape[0]), dtype=np.float32)
    nObs = len(indices)
    for i, idx in enumerate(indices):
        So[i, i] = So_d[i, 0]

    for i in tqdm(range(nObs)):
        time_val_i = H_Y_dict[indices[i]]['time']
        coord_i = (H_Y_dict[indices[i]]['ground_pixel'].centroid.y, H_Y_dict[indices[i]]['ground_pixel'].centroid.x)
        for j in range(i+1, nObs):
            coord_j = (H_Y_dict[indices[j]]['ground_pixel'].centroid.y, H_Y_dict[indices[j]]['ground_pixel'].centroid.x)
            time_val_j = H_Y_dict[indices[j]]['time']
            time_difference = np.abs((time_val_j - time_val_i).days)
            dist_val = np.abs(get_len(coord_i, coord_j))
            time_decay = np.exp(-time_difference/tau_time)
            dist_decay = np.exp(-dist_val/tau_space)
            sig_val = time_decay*dist_decay*np.sqrt(So_d[i]*So_d[j])
            if time_decay*dist_decay:
                So[i, j] = sig_val
                So[j, i] = sig_val

    return So