# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 13:48:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2024-06-23 22:44:17


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
import xarray as xr

import time
# from config import *

class getEmissions():
    def __init__(self, lons, lats, inventory_type, m, inventory_path=None, emissions=None):
        self.inventory_type = inventory_type
        self.lons = lons
        self.lats = lats
        self.path = inventory_path
        # self.date_range = date_range
        self.m = m

        print("Inventory type:", inventory_type)
        if not emissions:
            if self.inventory_type == "EDF":
                self.emissions = self.get_EDF_emissions()
            elif self.inventory_type == "EPA":
                self.emissions = self.get_EPA_emissions()
        else:
            self.emissions = emissions


    def new_grid(self, lat_axis, long_axis, cell_size_lon, cell_size_lat):
        """
        Creates desired grid from lat_min, lat_max, lon_min, lon_max, and a cell_size.

        Returns
        -------
        grid_cells
            List of shapely geometries representing the new grid boxes.
        """
        # Creates grid
        # lat_axis = np.arange(lat_min, lat_max, cell_size_lat)
        # long_axis = np.arange(lon_min, lon_max, cell_size_lon)
        grid_cells = [] # Where the regridded cells will be stored

        for x0 in long_axis:
            for y0 in lat_axis:
                # Bounds for each of the boxes
                x_1 = x0+cell_size_lon/2
                y_1 = y0+cell_size_lon/2
                x1 = x0-cell_size_lon/2
                y1 = y0-cell_size_lat/2
                grid_cells.append(shapely.geometry.box(x1, y1, x_1, y_1))
        
        return grid_cells

    def get_EDF_emissions(self):
        df = gpd.read_file(self.path)

        lons =self.lons
        lats = self.lats
        nrow = self.lats.shape[0]
        ncol = self.lons.shape[0]


        lon_min = lons[0]
        lon_max = lons[-1]
        lat_min = lats[0]
        lat_max = lats[-1]

        lon_res = lons[1] - lons[0]
        lat_res = lats[1] - lats[0]

        new_grid_interp = self.new_grid(lats, lons, lon_res, lat_res)
        gdf_grid = gpd.GeoDataFrame(new_grid_interp, geometry = new_grid_interp).drop(columns = 0)
        gdf_grid['centroid_lon'] = [val.centroid.x for val in gdf_grid['geometry']]
        gdf_grid['centroid_lat'] = [val.centroid.y for val in gdf_grid['geometry']]
        gdf_grid['Point'] = [Point(val.centroid.x, val.centroid.y) for val in gdf_grid['geometry']]
        gdf_grid = gdf_grid.set_crs("EPSG:4326")
        gdf_grid2 = gpd.GeoDataFrame(geometry=gdf_grid['Point'])
        gdf_grid2['box'] = gdf_grid['geometry']
        gdf_grid2 = gdf_grid2.set_crs("EPSG:4326")
        merged = gpd.sjoin(df, gdf_grid2, how='right')
        merged['mean_ch4_kgh'] = merged['mean_ch4_kgh'].fillna(0)
        merged['lower_bound_ch4_kgh'] = merged['lower_bound_ch4_kgh'].fillna(0)
        merged['upper_bound_ch4_kgh'] = merged['upper_bound_ch4_kgh'].fillna(0)

        # Computing Area of each grid
        # merged = merged.to_crs("EPSG:3857")
        area = np.array(merged['box'].to_crs("EPSG:3857").area).reshape(nrow, ncol, order='F')

        # Conversion to umol/m2/s

        conversion_factor = (1000)*(10**6)/(3600*16)
        ems_edf = np.array(merged['mean_ch4_kgh']).reshape(nrow, ncol, order='F')
        ems_edf_umolm2s = conversion_factor*ems_edf/area

        self.emissions = ems_edf_umolm2s
        return ems_edf_umolm2s

    def plot_emissions(self):
        h = plt.pcolor(self.lons, self.lats, self.emissions, vmin=0, vmax=5)
        plt.colorbar(h)
        plt.title(f"{self.inventory_type} inventory (umol/m2 s)")
        plt.show()


    def get_EPA_emissions(self):
        lons =self.lons
        lats = self.lats

        lon_min = lons[0]
        lon_max = lons[-1]
        lat_min = lats[0]
        lat_max = lats[-1]

        lon_res = lons[1] - lons[0]
        lat_res = lats[1] - lats[0]

        new_grid_interp = self.new_grid(lats, lons, lon_res, lat_res)
        gdf_grid = gpd.GeoDataFrame(new_grid_interp, geometry = new_grid_interp).drop(columns = 0)
        gdf_grid['centroid_lon'] = [val.centroid.x for val in gdf_grid['geometry']]
        gdf_grid['centroid_lat'] = [val.centroid.y for val in gdf_grid['geometry']]
        gdf_grid['Point'] = [Point(val.centroid.x, val.centroid.y) for val in gdf_grid['geometry']]
        gdf_grid = gdf_grid.set_crs("EPSG:4326")
        gdf_grid2 = gpd.GeoDataFrame(geometry=gdf_grid['Point'])
        gdf_grid2['box'] = gdf_grid['geometry']
        gdf_grid2 = gdf_grid2.set_crs("EPSG:4326")
        gdf_grid2['area'] = gdf_grid2['box'].to_crs("EPSG:3857").area
        area = np.array(gdf_grid2['area']).reshape(lats.shape[0], lons.shape[0], order='F')

        data = xr.open_dataset(self.path)

        Ao = 6.0221408e+23

        emission = np.zeros((350, 700))
        for var in data.variables:
            # print(var)
            if "emi" in var:
                emission += np.array(data[var][0])

        emission = np.multiply(emission, np.array(data['grid_cell_area'])[0])

        emission = emission*(3600*16)/(Ao*1000)  # molec per s to kg per hour conversion

        epa_lats = np.array(data['lat'])
        epa_lons = np.array(data['lon'])

        ems_epa = np.zeros((lats.shape[0], lons.shape[0]))
        for idx, lat in enumerate(lats):
            lat_index = np.abs(epa_lats - lat).argmin()
            for jdx, lon in enumerate(lons):
                lon_index = np.abs(epa_lons - lon).argmin()
                ems_epa[idx, jdx] = emission[lat_index, lon_index]


        conversion_factor = (1000)*(10**6)/(3600*16)  # Conversion to umol/s
        ems_epa_umolm2s = conversion_factor*ems_epa/area # Conversion to umol/m2/s
        self.emissions = ems_epa_umolm2s
        return ems_epa_umolm2s

    def compute_x_prior_vector(self, date_range):
        self.date_range = date_range
        Xa = np.zeros((self.date_range.shape[0]*self.m, 1), dtype=np.float32)
        for idx in tqdm(range(self.date_range.shape[0])):
            Xa[idx*self.m:(idx+1)*self.m] = self.emissions.reshape(-1, 1, order='F')
        return Xa