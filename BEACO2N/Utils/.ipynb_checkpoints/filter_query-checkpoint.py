# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-07-25 17:31:01
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 14:38:05


import pandas as pd
import datetime


def get_footprint_domain_df(footprint_files):
    """
        Identify the footprint domain from the footprint files

        Arguments:
            footprint_files: <list>
        returns:
            foot_df: <pandas dataframe>
    """
    footprint_list = []
    for file in footprint_files:
        [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = file.replace('.nc', '').split("_")
        receptor_lon, receptor_lat, receptor_agl = float(receptor_lon), float(receptor_lat), float(receptor_agl)
        year = int(timestamp[0:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = int(timestamp[8:])
        footprint_list.append([file, datetime.datetime(year, month, day, hour), receptor_lon, receptor_lat, receptor_agl])

    foot_df = pd.DataFrame(footprint_list, columns=['file', 'time', 'lon', 'lat', 'agl'])
    return foot_df

def get_emission_domain_df(emission_files):
    """
        Identify the emission domain from the emission files

        Arguments:
            emission_files: <list>
        returns:
            emission_df: <pandas dataframe>
    """
    emission_list = []
    for file in emission_files:
        trimmed_file = file.split("/")[-1].replace("_", "x")
        [_, year, month, day, hour] = trimmed_file.replace('.ncdf', '').split("x")
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        timestamp = datetime.datetime(year, month, day, hour)
        emission_list.append([file, timestamp])
        # break
    emission_df = pd.DataFrame(emission_list, columns=['file', 'time'])
    return emission_df

def filter_obs(files, lon_domain, lat_domain, agl_domain=''):
    """
        Filter the observations based on spatial domain

        Arguments:
            files: <list>
            lon_domain: <list>
            lat_domain: <list>
            agl_domain: <list>
        returns:
            <pandas series>, <pandas dataframe>
    """
    foot_df = get_footprint_domain_df(files)
    if not agl_domain:
        filtered_df = foot_df[(foot_df['lat']>=lat_domain[0])&(foot_df['lat']<lat_domain[1])&(foot_df['lon']>=lon_domain[0])&(foot_df['lon']<lon_domain[1])]
    elif agl_domain:
        filtered_df = foot_df[(foot_df['lat']>=lat_domain[0])&(foot_df['lat']<lat_domain[1])&(foot_df['lon']>=lon_domain[0])&(foot_df['lon']<lon_domain[1])&(foot_df['agl']>=agl_domain[0])&(foot_df['agl']<agl_domain[1])]
    
    return filtered_df['file'].values, filtered_df
        

def filter_emissions(files, time_domain):
    """
        Filter the emissions based on the time domain

        Arguments:
            files: <list>
            time_domain: <list>
        returns:
            <pandas series>, <pandas dataframe>
    """
    emission_df = get_emission_domain_df(files)
    filtered_df = emission_df[(emission_df['time']>=time_domain[0])&(emission_df['time']<time_domain[1])]
    return filtered_df['file'].values, filtered_df

