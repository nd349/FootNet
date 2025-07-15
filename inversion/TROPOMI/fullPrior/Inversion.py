# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 20:46:28
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-05-20 22:40:13


import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv
import time, datetime
import netCDF4 as nc
from tqdm import tqdm

# from config import *
from fullPrior.spatialPriorErrorCov import SpatialPriorCov
from fullPrior.temporalPriorErrorCov import compute_temporal_prior_error_covariance
from Utils.HQ_HQHT import HQ, HQHT

class InversionFullPrior():
    def __init__(self, H, Xp, Y, R, H_Y_dict, BKG, train_idx, tau_week, tau_day, ems_uncert, m, \
        spatial_covariance_path, nrow, ncol, start_time, end_time, temporal_frequency, emulator, \
        lons, lats, buffer_days, hq_parallel, location):
        self.H = csc_matrix(H)
        self.H_array = H
        self.Xp = csc_matrix(Xp)
        print("Dim of Xp:", self.Xp.shape)
        self.Y = csc_matrix(Y)
        self.R = csc_matrix(R)
        self.H_Y_dict = H_Y_dict
        self.buffer_days = buffer_days
        self.BKG = BKG
        self.train_idx = train_idx
        self.temporal_tau_week = tau_week
        self.temporal_tau_day = tau_day
        self.ems_uncert = ems_uncert
        self.m = m
        self.spatial_covariance_path = spatial_covariance_path
        self.nrow = nrow
        self.ncol = ncol
        self.start_time = start_time
        self.end_time = end_time
        self.temporal_frequency = temporal_frequency
        self.emulator = emulator
        self.lats = lats
        self.lons = lons
        self.hq_parallel = hq_parallel
        self.location = location

        self.D = compute_temporal_prior_error_covariance(Xp, self.temporal_tau_week, self.temporal_tau_day, self.ems_uncert, self.m)
        self.E = SpatialPriorCov().loadSa_xy(self.spatial_covariance_path, self.ems_uncert)


    def invert(self):
        start = time.time()

        mismatch = self.Y - csc_matrix.dot(self.H, self.Xp)
        HB = HQ(self.H_array, self.D, self.E, parallel=self.hq_parallel)
        HBHT = HQHT(HB, self.H_array, self.D, self.E)
        self.HBHT = HBHT.copy()
        G = HBHT + self.R
        G = csc_matrix(G)
        X_diff = csc_matrix.dot(HB.T, csc_matrix.dot(inv(G), mismatch))
        X_diff = X_diff.reshape(-1, 1)
        print("Dims of X_diff", X_diff.shape, type(X_diff))
        self.X_hat = self.Xp + X_diff
        self.X_hat = csc_matrix(self.X_hat)
        print(f"Time taken for inversion: {time.time()-start} seconds")
        self.X_post = self.remove_padding(self.X_hat)
        return self.X_hat.toarray()


    def remove_padding(self, X_hat):
        """
        Removes padding from the solution (back hours)

        Arguments:
            X_hat: <1-D array>
        returns:
            X_hat: <1-D array>
        """
        # import pdb; pdb.set_trace()
        X_hat = csc_matrix(X_hat)
        print("Type of X_hat:", type(X_hat))
        # X_hat = X_hat[back_hours*m:(X_hat.shape[0]-back_hours*m)]
        X_week = X_hat.toarray()[self.buffer_days*self.m:(X_hat.shape[0]-self.buffer_days*self.m)]
        X_post = np.zeros((int(X_week.shape[0]/self.m), self.nrow, self.ncol))
        for idx in range(int(X_week.shape[0]/self.m)):
            X_post[idx, :, :] = X_week[idx*self.m:(idx+1)*self.m].reshape(self.nrow, self.ncol, order='F')
        return X_post

    def get_concentrations(self, H, X):
        return np.array(np.dot(H, X))

    def save_concentrations(self, output_directory, H_valid=None, Y_valid=None, BKG_valid=None, valid_idx=None, cross_validation=False, cross_validation_fraction=None):
        """
        Saving posterior solution

        Arguments:
            None
        returns:
            None
        """

        self.cross_validation = cross_validation
        self.cross_validation_fraction = cross_validation_fraction

        H_train = self.H_array
        X_pri = self.Xp.toarray()
        X_hat = self.X_hat.toarray()

        y_posterior_train = self.get_concentrations(H_train, X_hat)
        y_prior_train = self.get_concentrations(H_train, X_pri)

        # y_posterior_train, y_prior_train = self.get_concentrations()
        year = str(self.start_time.year)
        month = str(self.start_time.month)
        day = str(self.start_time.day)
        

        if len(month) == 1:
            month = "0"+month
        if len(day) == 1:
            day = "0"+day
        

        if self.emulator:
            conc_file = f"{output_directory}emulator_observations_prior_posterior_{year}x{month}x{day}.nc"
        else:
            conc_file = f"{output_directory}XSTILT_observations_prior_posterior_{year}x{month}x{day}.nc"

        conc_nc = nc.Dataset(conc_file, "w", format="NETCDF4")
        conc_nc.createDimension("nobs", self.Y.shape[0])
        obs_prior_train = conc_nc.createVariable("obs_prior_train", "f8", ("nobs"))
        obs_posterior_train = conc_nc.createVariable("obs_posterior_train", "f8", ("nobs"))
        obs_actual_train = conc_nc.createVariable("obs_actual_train", "f8", ("nobs"))

        obs_prior_train[:] = y_prior_train
        obs_posterior_train[:] = y_posterior_train
        obs_actual_train[:] = self.Y.toarray()

        obs_year = conc_nc.createVariable("obs_year", "f8", ("nobs"))
        obs_month = conc_nc.createVariable("obs_month", "f8", ("nobs"))
        obs_day = conc_nc.createVariable("obs_day", "f8", ("nobs"))
        obs_hour = conc_nc.createVariable("obs_hour", "f8", ("nobs"))

        obs_lats = conc_nc.createVariable("obs_lat_train", "f8", ("nobs"))
        obs_lons = conc_nc.createVariable("obs_lon_train", "f8", ("nobs"))

        obs_year[:] = [datetime.datetime.strftime(self.H_Y_dict[term]['time'], '%Y%m%d%H')[:4] for term in self.train_idx]
        obs_month[:] = [datetime.datetime.strftime(self.H_Y_dict[term]['time'], '%Y%m%d%H')[4:6] for term in self.train_idx]
        obs_day[:] = [datetime.datetime.strftime(self.H_Y_dict[term]['time'], '%Y%m%d%H')[6:8] for term in self.train_idx]
        obs_hour[:] = [datetime.datetime.strftime(self.H_Y_dict[term]['time'], '%Y%m%d%H')[8:] for term in self.train_idx]

        obs_lats[:] = [self.H_Y_dict[term]['ground_pixel'].centroid.y for term in self.train_idx]
        obs_lons[:] = [self.H_Y_dict[term]['ground_pixel'].centroid.x for term in self.train_idx]

        conc_nc.createVariable("background_train", "f8", ("nobs"))[:] = self.BKG

        conc_nc.createVariable("HBHT", "f8", ("nobs", "nobs"))[:, :] = self.HBHT
        conc_nc.createVariable("R", "f8", ("nobs", "nobs"))[:, :] = self.R.toarray()


        # Cross validation
        if self.cross_validation:
            y_posterior_valid = self.get_concentrations(H_valid, X_hat)
            y_prior_valid = self.get_concentrations(H_valid, X_pri)

            conc_nc.createDimension("nobs_valid", Y_valid.shape[0])
            conc_nc.createDimension("info", 1)
            conc_nc.createVariable("obs_prior_valid", "f8", ("nobs_valid"))[:] = y_prior_valid
            conc_nc.createVariable("obs_posterior_valid", "f8", ("nobs_valid"))[:] = y_posterior_valid
            conc_nc.createVariable("obs_actual_valid", "f8", ("nobs_valid"))[:] = Y_valid

            conc_nc.createVariable("obs_lat_valid", "f8", ("nobs_valid"))[:] = [self.H_Y_dict[term]['ground_pixel'].centroid.y for term in valid_idx]
            conc_nc.createVariable("obs_lon_valid", "f8", ("nobs_valid"))[:] = [self.H_Y_dict[term]['ground_pixel'].centroid.x for term in valid_idx]
            conc_nc.createVariable("cross_validation_fraction", "f8", ("info"))[:] = self.cross_validation_fraction

            conc_nc.createVariable("background_valid", "f8", ("nobs_valid"))[:] = BKG_valid

        # Saving cumulative influence of footprints
        conc_nc.createDimension("lat", self.nrow)
        conc_nc.createDimension("lon", self.ncol)
        lat = conc_nc.createVariable("lat", "f8", ("lat",))
        lon = conc_nc.createVariable("lon", "f8", ("lon",))
        
        lat[:] = self.lats
        lon[:] = self.lons

        cum_foot = np.zeros((self.nrow, self.ncol))
        for key in self.H_Y_dict:
            gp_foot = self.H_Y_dict[key]['gp_foot']
            cum_foot += np.nansum(gp_foot, axis=0)

        foot_influence = conc_nc.createVariable("foot_influence", "f8", ("lat", "lon"))
        foot_influence[:, :] = cum_foot
        
        conc_nc.close()
        print(f"{conc_file} has been saved ....")


    def save_solution(self, output_directory):
        """
        Saving posterior solution

        Arguments:
            None
        returns:
            None
        """
        X_hat = self.X_hat
        X_hat_grid = self.X_post
        
        solution_date_range = pd.date_range(start=self.start_time, end=self.end_time, freq=self.temporal_frequency)

        print("Saving output at:", output_directory)
        

        for idx, timestamp in tqdm(enumerate(solution_date_range)):
            year = str(timestamp.year)
            month = str(timestamp.month)
            day = str(timestamp.day)
            # hour = str(timestamp.hour)
            if len(month)==1:
                month = '0'+month
            if len(day) == 1:
                day = '0'+day
            # if len(hour) == 1:
            #     hour = '0'+hour
            timestamp = f"{year}{month}{day}"
            if self.emulator:    
                file = f"{output_directory}FootNet_posterior_{year}x{month}x{day}.ncdf"
            else:
                file = f"{output_directory}XSTILT_posterior_{year}x{month}x{day}.ncdf"
            flux = X_hat_grid[idx, :, :]

            # import pdb; pdb.set_trace()
            out_nc = nc.Dataset(file, "w", format='NETCDF4')
            out_nc.createDimension("lat", self.nrow)
            out_nc.createDimension("lon", self.ncol)
            out_nc.createDimension("info", 1)
            
            lat = out_nc.createVariable("lat", "f8", ("lat",))
            lon = out_nc.createVariable("lon", "f8", ("lon",))
            
            lat[:] = self.lats
            lon[:] = self.lons
            
            soln = out_nc.createVariable("flux_umol_m2_s", "f8", ("lat", "lon"))
            soln[:,:] = flux

            out_nc.close()
            print(f"{file} has been saved ...")









