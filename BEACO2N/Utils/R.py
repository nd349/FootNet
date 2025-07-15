# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-09-06 13:21:53
# @Last Modified by:   nd349
# @Last Modified time: 2024-09-06 13:35:51

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from tqdm import tqdm

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

def fill_observation_covariance_matrix(So, So_d, lowBound=1e-5, tau_time=1, tau_space=2):
    """
        Compute the observational covariance matrix

        Arguments:
		So: <numpy array>  Observation Error Covariance matrix with zero values (nobs, nobs) # So = np.zeros((So_d.shape[0], So_d.shape[0]), dtype=np.float32)
		So_d: <numpy array>  Diagonal terms of So (nobs, 1)
		lowBound: <float>
		tau_time: <int/float>
		tau_space: <int/float>
        returns:
        So: <numpy array>    
    """
    
    print("Forming observation covariance matrix ....")
    nObs = So_d.shape[0]
    
    # Adding diagonal terms
    for i in range(nObs):
        So[i, i] = So_d[i, 0]  # Filling diagonal terms of So

    # Adding off-diagonal terms
    for i in tqdm(range(nObs)):
    	# observation_dict is a global parameter which contains obs info such as time and location of obs sites
        time_val_i = observation_dict[i]['time']  # Time of obsi
        coord_i = (observation_dict[i]['lat'], observation_dict[i]['lon']) # Location of obsi
        for j in range(i+1, nObs):
            coord_j = (observation_dict[j]['lat'], observation_dict[j]['lon']) # Location of obsj
            time_val = (observation_dict[j]['time']-time_val_i).seconds/3600 # Time lag between obsj and obsi
            dist_val = get_len(coord_i, coord_j) # Distance between coord_i and coord_j
            time_decay = np.exp(-abs(time_val)/tau_time) # Computing time decay term, tau_time is in hours
            dist_decay = np.exp(-abs(dist_val)/tau_space) # Computing distance decay term, tau_space is in km
            sig_val = time_decay*dist_decay*np.sqrt(So_d[i]*So_d[j]) # Combining everything together
            if time_decay*dist_decay > lowBound: # lowBound is a threshold (1e-5 in our case)
                So[i, j] = sig_val
                So[j, i] = sig_val
    So = csc_matrix(So) # Converting it into sparse matrix for faster computation efficiency.
    return So