# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-28 19:19:58
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-09-30 14:33:32

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv
import torch, time
import netCDF4 as nc
from tqdm import tqdm
from config import *
from fullCovariance.SpatialCovariance import *
from fullCovariance.TemporalCovariance import *
from Utils.HQ_HQHT import HQ, HQHT
from Utils.reshape_matrices import *



class InversionFullPrior():
    def __init__(self, H, X, Y, So):
        self.mode = mode
        self.full_prior = full_prior
        self.sparse = sparse
        self.ems_uncert = ems_uncert
        self.minUncert = minUncert
        self.device = device
        self.tau_day = tau_day
        self.tau_hr = tau_hr
        self.tau_len = tau_len
        self.fsigma = fsigma
        self.Sa_xy_file = Sa_xy_file
        self.output_file = output_posterior_file
        if not self.full_prior:
            raise Exception(f"diag_prior is expected to be True but found this instead: {diag_prior}")

        if self.sparse:
            print("Forming sparse matrices ....")
            self.H = csc_matrix(H)
            self.H_array = H
            self.X_pri = csc_matrix(X)
            self.X_pri_array = X
            self.Y = csc_matrix(Y)
            self.Sa_t = build_temporal(self.tau_day, self.tau_hr, self.X_pri_array, m) # numpy array
        else:
            self.H = H
            self.X_pri = X
            self.Y = Y
            self.Sa_t = build_temporal(self.tau_day, self.tau_hr, self.X_pri_array, m)

        # self.Sa_xy = self.form_spatial_covariance()
        self.Sa_xy = self.form_spatial_covariance(X) # csr array or csc matrix?
        
        self.So = So # csc matrix
        self.X_hat = None

        print(f"Type of H:{type(self.H)}")
        print(f"Type of X :{type(self.X_pri)}")
        print(f"Type of Y:{type(self.Y)}")
        print(f"Type of Sa_t:{type(self.Sa_t)}")
        print(f"Type of Sa_xy:{type(self.Sa_xy)}")
        print(f"Type of So:{type(self.So)}")



    # def form_temporal_covariance(self):
    #     Sa_t, variance_t = build_temporal(self.tau_day, self.tau_hr, self.X_pri_array, m)
    #     Sa_t = np.sqrt(self.fsigma)*(np.dot(np.sqrt(variance_t), np.dot(Sa_t, np.sqrt(variance_t))))
    #     return Sa_t

    def form_spatial_covariance(self, X):
        # import pdb; pdb.set_trace()
        print ("Forming spatial covariance")
        Sa_xy = load_Sa_xy(self.Sa_xy_file)

        nEms = int(self.X_pri.shape[0]/m)
        nG = m
        emsAll = np.zeros((nEms, nG), dtype=np.float32)
        variance_xy = np.zeros((nG, nG), dtype=np.float32)
        for i in tqdm(range(nG)):
            emsAll[:, i] = np.array([X[j*m:(j+1)*m, 0][i] for j in range(nEms)])
            variance_xy[i, i] = np.var(emsAll[:, i])

        print("Converting variance into sparse")
        variance_xy = csc_matrix(variance_xy)
        variance_xy_sqrt = csc_matrix.sqrt(variance_xy)
        print("Computing covariance matrix")
        Sa_xy_covariance = np.sqrt(self.fsigma)*(variance_xy_sqrt.multiply(Sa_xy.multiply(variance_xy_sqrt)))
        # Sa_xy = np.sqrt(self.fsigma)*(csr_matrix.dot(csr_matrix.sqrt(variance_xy), csr_matrix.dot(Sa_xy), csr_matrix.sqrt(variance_xy)))
        return Sa_xy_covariance

    def invert(self):
        print("Inversion is starting .....")
        print(f"Size of H: {self.H.shape}")
        print(f"Size of X prior: {self.X_pri.shape}")
        print(f"Size of Y: {self.Y.shape}")
        print(f"Size of So: {self.So.shape}")
        print(f"Size of Sa_t: {self.Sa_t.shape}")
        print(f"Size of Sa_xy: {self.Sa_xy.shape}")


        start = time.time()
        if self.device == 'cpu' and self.sparse:
            print("computing mismatch ....")
            mismatch = self.Y - csc_matrix.dot(self.H, self.X_pri)
            print("computing HQ ....")
            KSa = HQ(self.H_array, self.Sa_t, self.Sa_xy)
            print("computing HQHT ....")
            G = HQHT(KSa, self.H_array, self.Sa_t, self.Sa_xy)
            print("computing gain matrix ....")
            G = G + self.So
            G = csc_matrix(G)
            mismatch1 = csc_matrix.dot(inv(G), mismatch)
            X_dif = csc_matrix.dot(KSa.T, mismatch1)
            print("computing posterior emissions ....")
            X_hat = self.X_pri + X_dif
            self.original_X_hat = X_hat
            self.X_hat = self.remove_padding(X_hat)
            print(f"Time taken for inversion: {time.time()-start} seconds")
            return self.X_hat

        elif self.device == 'cuda':
            pass



    def remove_padding(self, X_hat):
        X_hat = X_hat[back_hours*m:(X_hat.shape[0]-back_hours*m)]
        return X_hat


    def save_solution(self):
        X_hat = self.X_hat
        X_hat_grid = np.zeros((int(X_hat.shape[0]/m), nrow, ncol))
        solution_date_range = pd.date_range(start=start_time, end=end_time-datetime.timedelta(hours=1), freq='1h')
        
        for idx in range(X_hat_grid.shape[0]):
            # X_hat_grid[idx, :, :] = make_grid_2d_column(X_hat[idx*m:(idx+1)*m, 0], nrow, ncol)
            X_hat_grid[idx, :, :] = X_hat[idx*m:(idx+1)*m, 0].reshape(nrow, ncol, order='F')
        
        for idx, timestamp in tqdm(enumerate(solution_date_range)):
            year = str(timestamp.year)
            month = str(timestamp.month)
            day = str(timestamp.day)
            hour = str(timestamp.hour)
            if len(month)==1:
                month = '0'+month
            if len(day) == 1:
                day = '0'+day
            if len(hour) == 1:
                hour = '0'+hour
            timestamp = f"{year}{month}{day}{hour}"
            file = f"{output_directory}{location}_{year}x{month}x{day}x{hour}.ncdf"
            flux = X_hat_grid[idx, :, :]
            print(file)


            out_nc = nc.Dataset(file, "w", format='NETCDF4')
            out_nc.createDimension("lat", nrow)
            out_nc.createDimension("lon", ncol)
            out_nc.createDimension("info", 1)
            lat = out_nc.createVariable("lat", "f8", ("lat",))
            lon = out_nc.createVariable("lon", "f8", ("lon",))

            lat[:] = lats
            lon[:] = lons
            
            soln = out_nc.createVariable("flux", "f8", ("lat", "lon"))
            soln[:,:] = flux
            out_nc.close()






