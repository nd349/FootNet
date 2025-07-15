# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 20:46:28
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2024-06-23 22:21:48

import numpy as np
import pickle
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from joblib import Parallel, delayed
# from config import *


class SpatialPriorCov():
    def __init__(self):
        pass

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


    def fill_Sa_xy_without_uncertainty(self, i):
        rows_saxy = []
        cols_saxy = []
        data_saxy = []

        sigmai = self.emsAll[i]
        for j in range(i, self.nG):
            sigmaj = self.emsAll[j]
            distance = self.get_len(self.grid_flattened[i], self.grid_flattened[j])
            if distance < self.minimum_distance:
                dist_decay = np.exp(-abs(distance)/self.tau_len)
                sig_val = np.sqrt(sigmai*sigmaj)*dist_decay
                rows_saxy.append(i)
                cols_saxy.append(j)
                rows_saxy.append(j)
                cols_saxy.append(i)
                data_saxy.append(sig_val)
                data_saxy.append(sig_val)
        return rows_saxy, cols_saxy, data_saxy

    def computeSa_xy(self, Xa, ems_uncert, lats, lons, tau_len, min_distance, inventory_type, spatial_covariance_path):
        self.grid_flattened = [(lat, lon) for lon in lons for lat in lats]
        # self.Xa = Xa
        self.ems_uncert = ems_uncert
        self.minimum_distance = min_distance
        self.tau_len = tau_len
        self.nG = lats.shape[0]*lons.shape[0]
        self.inventory_type = inventory_type
        self.spatial_covariance_path = spatial_covariance_path

        self.emsAll = Xa[:m][:, 0]
        self.nEms = int(Xa.shape[0]/m)

        print("Constructing off diagonal spatial covariance without ems uncertainty ...")
        OUTPUT = Parallel(n_jobs=128, verbose=10, backend='multiprocessing')(delayed(self.fill_Sa_xy_without_uncertainty)(i) for i in range(self.nG))

        rows = []
        cols = []
        data = []

        for value in OUTPUT:
            rows += value[0]
            cols += value[1]
            data += value[2]

        Sa_xy = csr_matrix((data, (rows, cols)), 
                                  shape = (self.nG, self.nG), dtype=np.float32)

        Sa_xy = csc_matrix(Sa_xy)
        outfile = self.spatial_covariance_path
        with open(outfile, "wb") as file:
            print(f"Writing spatial prior error covariance at {outfile} ... Make sure that emission uncertainty is applied to it before use.")
            pickle.dump(Sa_xy, file)

    def loadSa_xy(self, outfile, ems_uncert):
        print(f"Loading off diagonal spatial covariance matrix and multiplying with ems uncertainty {ems_uncert} ...")
        # outfile = self.spatial_covariance_path
        with open(outfile, "rb") as file:
            Sa_xy = pickle.load(file)

        Sa_xy = Sa_xy*ems_uncert
        # print(np.min(Sa_xy), np.max(Sa_xy), np.average(Sa_xy))
        return Sa_xy

    

