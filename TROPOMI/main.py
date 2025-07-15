# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2025-06-11 09:05:39

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import geopandas as gpd
import glob, time, shapely
import datetime, os
import netCDF4 as nc
import sklearn.metrics as metrics
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely import Polygon, Point
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from config import *
from Utils.load_H_Y_dict import *
from Utils.fillJacobianH import compute_H
from Utils.getEnhancements import compute_Y_So_d, compute_Y_valid
from Utils.emissions import getEmissions
from Utils.getObsCovariance import compute_obs_covariance
from DiagPrior.diagInversion import diagPriorInversion
from fullPrior.Inversion import InversionFullPrior
from FootNet.FootNet import FootNet
from FootNet.get_meteorology_footnet import input_meteorology

def plot_results(output_directory, start_time):
    start_time = datetime.datetime.strftime(start_time, "%Y%m%d%H")
    diag = [i for i in range(-90, 90)]
    if cross_validation:
        fig, ax = plt.subplots(1, 5, figsize=(30, 5))
        ax[4].scatter(Y_valid, csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0])
        ax[4].plot(diag, diag, '--', color='k')
        ax[4].set_xlabel("Actual enhancements", fontsize=15)
        ax[4].set_ylabel("Model simulated enhancements", fontsize=15)
        ax[4].set_title("Posterior mismatch (validation)", fontsize=15)
        ax[4].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0], Y_valid[:, 0])[0]), 3)}", (20, 0), fontsize=10)
        ax[4].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0], Y_valid[:, 0])), 3)}", (20, -15), fontsize=10)
        ax[4].set_xlim([-20, 60])
        ax[4].set_ylim([-20, 60])
    else:
        fig, ax = plt.subplots(1, 4, figsize=(25, 5))

    h = ax[0].pcolor(lons, lats, np.average(X_post, axis=0), vmin=0, vmax=10)
    fig.colorbar(h, ax=ax[0])
    ax[0].set_title(f"Posterior ({ems_uncert*100}%)", fontsize=15)

    h = ax[1].pcolor(lons, lats, np.average(X_post, axis=0) - ems.emissions, vmin=-1e-1, vmax=1e-1)
    fig.colorbar(h, ax=ax[1])
    ax[1].set_title(f"Posterior - Prior ({ems_uncert*100}%)", fontsize=15)


    
    ax[2].scatter(inversion.Y.toarray()[:, 0], csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0])
    ax[2].plot(diag, diag, '--', color='k')
    ax[2].set_xlabel("Actual enhancements", fontsize=15)
    ax[2].set_ylabel("Model simulated enhancements", fontsize=15)
    ax[2].set_title(f"Prior mismatch ({ems.inventory_type})", fontsize=15)
    ax[2].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0], inversion.Y.toarray()[:, 0])[0]), 3)}", (20, 0), fontsize=10)
    ax[2].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0], inversion.Y.toarray()[:, 0])), 3)}", (20, -15), fontsize=10)
    ax[2].set_xlim([-20, 60])
    ax[2].set_ylim([-20, 60])

    ax[3].scatter(inversion.Y.toarray()[:, 0], csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0])
    ax[3].plot(diag, diag, '--', color='k')
    ax[3].set_xlabel("Actual enhancements", fontsize=15)
    ax[3].set_ylabel("Model simulated enhancements", fontsize=15)
    ax[3].set_title(f"Posterior mismatch ({ems.inventory_type})", fontsize=15)
    ax[3].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0], inversion.Y.toarray()[:, 0])[0]), 3)}", (20, 0), fontsize=10)
    ax[3].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0], inversion.Y.toarray()[:, 0])), 3)}", (20, -15), fontsize=10)
    ax[3].set_xlim([-20, 60])
    ax[3].set_ylim([-20, 60])

    fig.suptitle(f"Flux Inversion Using Diagonal Prior ({ems.inventory_type})", fontsize=15)

    if off_diag and cross_validation:
        savefile = f"{output_directory}{start_time}_sample_flux_off_diag_cross_validation_{ems.inventory_type}_{ems_uncert}.png"
    elif off_diag and not cross_validation:
        savefile = f"{output_directory}{start_time}_sample_flux_off_diag_{ems.inventory_type}_{ems_uncert}.png"
    elif not off_diag and cross_validation:
        savefile = f"{output_directory}{start_time}_sample_flux_diag_cross_validation_{ems.inventory_type}_{ems_uncert}.png"
    else:
        savefile = f"{output_directory}{start_time}_sample_flux_diag_{ems.inventory_type}_{ems_uncert}.png"
    fig.savefig(savefile)



if not os.path.exists(spatial_covariance_path):
    from fullPrior.spatialPriorErrorCov import SpatialPriorCov
    ems = getEmissions()
    Xp = ems.compute_x_prior_vector()
    spatial_cov = SpatialPriorCov()
    spatial_cov.computeSa_xy(Xp)


domain_gdf = get_larger_domain_obs_background(domain_obs)
df = get_obs(obs_file)
dk = get_obs_time_domain(df, m_start, m_end)

background = background(dk, domain_gdf, HRRR_DIR, trimsize, lons, lats, hr3lat_full, hr3lon_full, octant_dict, background_date_range)
readFoot = ReadFootprints(transport_max_backhours, nrow, ncol, transport_max_days, clon_index=clon_index, clat_index=clat_index)

if foot_model == "FootNet":
    input_met = input_meteorology(list(set(dk['time'])), lons, lats, trimsize, hr3lat_full, hr3lon_full, HRRR_DIR, footnet_hours_mode, maximum_domain_trajectory)
    unet = FootNet(footnet, device, n_channels, n_classes, emulator_model_path, epsilon)
    H_Y_dict = get_H_Y_dict(dk, readFoot, foot_model, background, mode, footnet_model=unet, input_met=input_met, batch_size=batch_size, num_workers=num_workers, transport_max_backhours=transport_max_backhours)
else:
    H_Y_dict = get_H_Y_dict(dk, readFoot, foot_model, background, mode, num_workers=num_workers)

if cross_validation:
    train_idx, valid_idx = train_test_split(list(H_Y_dict.keys()), test_size=cross_validation_fraction, random_state=42)
    Y, So_d, BKG = compute_Y_So_d(H_Y_dict, train_idx)
    Y_valid, BKG_valid = compute_Y_valid(H_Y_dict, valid_idx)
    H = compute_H(H_Y_dict, train_idx, date_range, m, time_dict)
    H_valid = compute_H(H_Y_dict, valid_idx, date_range, m, time_dict)
    R = compute_obs_covariance(So_d, obs_tau_time, obs_tau_space, H_Y_dict, train_idx)
else:
    train_idx = list(H_Y_dict.keys())
    Y, So_d, BKG = compute_Y_So_d(H_Y_dict, train_idx)
    R = compute_obs_covariance(So_d, obs_tau_time, obs_tau_space, H_Y_dict, train_idx)
    H = compute_H(H_Y_dict, train_idx, date_range, m, time_dict)

ems = getEmissions(lons, lats, inventory_type, m, inventory_path=inventory_path)
Xp = ems.compute_x_prior_vector(date_range)

print(Y)
print(H)
print(Xp)
print(ems.emissions)
print(R)


if __name__== '__main__':
    # uncert_list = [50/100, 100/100, 150/100, 200/100, 250/100, 300/100, 350/100, 400/100, 450/100, 500/100, 550/100, 600/100, \
    # 650/100, 700/100, 750/100, 800/100, 850/100, 900/100, 950/100, 1000/100]
    uncert_list = [50/100, 100/100, 250/100, 500/100, 1000/100]
    for ems_uncert in uncert_list:
        print("Emission uncertainty:", ems_uncert)
        output_directory = location + "ems_uncert" + str(ems_uncert)+"/"
        print(f"output directory: {output_directory}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if off_diag:
            inversion = InversionFullPrior(H, Xp, Y, R, H_Y_dict, BKG, train_idx, tau_week, tau_day, ems_uncert, m, \
            spatial_covariance_path, nrow, ncol, start_time, end_time, temporal_frequency, emulator, lons, lats, \
            buffer_days, hq_parallel, location)
        else:
            inversion = diagPriorInversion(H, Xp, Y, R, H_Y_dict)
        X_hat = inversion.invert()
        X_post = inversion.X_post
        if cross_validation:
            inversion.save_concentrations(output_directory, H_valid, Y_valid, BKG_valid, valid_idx, \
                cross_validation=cross_validation, cross_validation_fraction=cross_validation_fraction)
        else:
            inversion.save_concentrations(output_directory)
        inversion.save_solution(output_directory)

        # plot_results(output_directory, start_time)
    # diag = [i for i in range(-20, 50)]
    # if cross_validation:
    #     fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    #     ax[4].scatter(Y_valid, csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0])
    #     ax[4].plot(diag, diag, '--', color='k')
    #     ax[4].set_xlabel("Actual enhancements", fontsize=15)
    #     ax[4].set_ylabel("Model simulated enhancements", fontsize=15)
    #     ax[4].set_title("Posterior mismatch (validation)", fontsize=15)
    #     ax[4].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0], Y_valid[:, 0])[0]), 3)}", (20, 0), fontsize=10)
    #     ax[4].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(csc_matrix(H_valid), inversion.X_hat).toarray()[:, 0], Y_valid[:, 0])), 3)}", (20, -15), fontsize=10)

    # else:
    #     fig, ax = plt.subplots(1, 4, figsize=(25, 5))

    # h = ax[0].pcolor(lons, lats, np.average(X_post, axis=0), vmin=0, vmax=10)
    # fig.colorbar(h, ax=ax[0])
    # ax[0].set_title(f"Posterior ({ems_uncert*100}%)", fontsize=15)

    # h = ax[1].pcolor(lons, lats, np.average(X_post, axis=0) - ems.emissions, vmin=-1e-1, vmax=1e-1)
    # fig.colorbar(h, ax=ax[1])
    # ax[1].set_title(f"Posterior - Prior ({ems_uncert*100}%)", fontsize=15)


    
    # ax[2].scatter(inversion.Y.toarray()[:, 0], csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0])
    # ax[2].plot(diag, diag, '--', color='k')
    # ax[2].set_xlabel("Actual enhancements", fontsize=15)
    # ax[2].set_ylabel("Model simulated enhancements", fontsize=15)
    # ax[2].set_title(f"Prior mismatch ({ems.inventory_type})", fontsize=15)
    # ax[2].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0], inversion.Y.toarray()[:, 0])[0]), 3)}", (20, 0), fontsize=10)
    # ax[2].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(inversion.H, inversion.Xp).toarray()[:, 0], inversion.Y.toarray()[:, 0])), 3)}", (20, -15), fontsize=10)

    # ax[3].scatter(inversion.Y.toarray()[:, 0], csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0])
    # ax[3].plot(diag, diag, '--', color='k')
    # ax[3].set_xlabel("Actual enhancements", fontsize=15)
    # ax[3].set_ylabel("Model simulated enhancements", fontsize=15)
    # ax[3].set_title(f"Posterior mismatch ({ems.inventory_type})", fontsize=15)
    # ax[3].annotate(f"r = {round(float(pearsonr(csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0], inversion.Y.toarray()[:, 0])[0]), 3)}", (20, 0), fontsize=10)
    # ax[3].annotate(f"MSE = {round(float(metrics.mean_squared_error(csc_matrix.dot(inversion.H, inversion.X_hat).toarray()[:, 0], inversion.Y.toarray()[:, 0])), 3)}", (20, -15), fontsize=10)

    # fig.suptitle(f"Flux Inversion Using Diagonal Prior ({ems.inventory_type})", fontsize=15)

    # if off_diag and cross_validation:
    #     savefile = f"sample_flux_off_diag_cross_validation_{ems.inventory_type}_{ems_uncert}.png"
    # elif off_diag and not cross_validation:
    #     savefile = f"sample_flux_off_diag_{ems.inventory_type}_{ems_uncert}.png"
    # elif not off_diag and cross_validation:
    #     savefile = f"sample_flux_diag_cross_validation_{ems.inventory_type}_{ems_uncert}.png"
    # else:
    #     savefile = f"sample_flux_diag_{ems.inventory_type}_{ems_uncert}.png"
    # fig.savefig(savefile)

