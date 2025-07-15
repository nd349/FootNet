# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:59:27
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2024-06-16 14:34:41

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from Utils.readData import *
from Utils.filter_query import *
from Utils.reshape_matrices import *
# import emulator.footnet_multiple as ftnet
from config import *
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split

print(f"Location: {location}")
print(f"Emulator:", emulator)
print(f"Emulator run:", emulator_run)
print(f"Mode: {mode}")
print(f"Device: {device}")
print(f"Start date: {start_time}")
print(f"End date: {end_time}")
print(f"back_hours: {back_hours}")
print(f"m_start: {m_start}")
print(f"m_end: {m_end}")
print(f"Inversion grid: {Inv_lonLim, Inv_latLim}")
# print(f"Posterior solutions output location: {output_posterior_file}")
print(f"Output directory: {output_directory}")
print()

footprint_files = get_files(footprint_directory)
emission_files = get_files(emission_directory)
emission_files.sort()
footprint_files.sort()
print(len(footprint_files), len(emission_files))

emission_filtered_files, emission_files_df = filter_emissions(emission_files, [m_start, m_end])
foot_filtered_files, foot_files_df = filter_obs(footprint_files, Inv_lonLim, Inv_latLim, agl_domain='')

foot_files_time_df = foot_files_df[(foot_files_df['time']>=start_time)&(foot_files_df['time']<end_time)]
# print(foot_files_time_df)


global H_valid, Y_valid, validation_dict, H_check_valid, BKG_valid
global H, Y, So_d, X, observation_dict, So, BKG

global H_check

if cross_validation:
    domain_foot_files_train, domain_foot_files_validation = train_test_split(foot_files_time_df, test_size=cross_validation_fraction, random_state=42)
    domain_foot_files = list(domain_foot_files_train['file'])
    domain_foot_files_valid = list(domain_foot_files_validation['file'])
    validation_dict = {}
    H_valid = np.zeros((len(domain_foot_files_valid), date_range.shape[0]*m), dtype=np.float32)
    H_check_valid = np.zeros((len(domain_foot_files_valid), date_range.shape[0]*m), dtype=np.float32)
    Y_valid = np.zeros((len(domain_foot_files_valid), 1), dtype=np.float32)
    BKG_valid = np.zeros((len(domain_foot_files_valid), 1), dtype=np.float32)
    print("Training size:", len(domain_foot_files))
    print("Validation size:", len(domain_foot_files_valid))
    # Update the domain_foot_files
    # Further, we also have to declare and load H_cross_validation and X_cross_validation
    pass
else:
    print("Observations have not been kept out for the validation ....(everything is used in the training)")
    domain_foot_files = list(foot_files_time_df['file'])
    pass # No updates in domain_foot_files




observation_dict = {}
H = np.zeros((len(domain_foot_files), date_range.shape[0]*m), dtype=np.float32)
H_check = np.zeros((len(domain_foot_files), date_range.shape[0]*m), dtype=np.float32)
Y = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
BKG = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
So_d = np.zeros((len(domain_foot_files), 1), dtype=np.float32)
X = np.zeros((date_range.shape[0]*m, 1), dtype=np.float32)
nObs = So_d.shape[0]
So = np.zeros((nObs, nObs), dtype=np.float32)
receptor_list = []

def get_background_error(data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux):
    """
        Get the background error

        Arguments:
            data: <netcdf object>
            bkg_conc_noaa: <float>
            bkg_conc_nasa: <float>
            bkg_conc_ameriflux: <float>
            bkg_err_noaa: <float>
            bkg_err_nasa: <float>
            bkg_err_ameriflux: <float>
        returns:
            <float>, <float>, <str>
    """
    source = ''
    bkg_conc = bkg_conc_noaa
    bkg_err = max(bkg_err_nasa, bkg_err_noaa)
    if bkg_err_nasa > bkg_err_noaa:
        source = 'nasa'
    else:
        source = 'noaa'
    if bkg_err == -999.0: #nan value auto fill in netcdf file
        if bkg_err_nasa == -999.0 and bkg_err_noaa == -999.0:
            bkg_err = np.nan
        elif bkg_err_nasa == -999.0:
            source = 'noaa'
            bkg_err = bkg_err_noaa
        elif big_err_noaa == -999.0:
            source = 'nasa'
            bkg_err = bkg_err_nasa
    
    ameriflux_lon = -121.8
    ameriflux_lat =   38.2
    maxDist       =   25.0 # maximum allowable distance from the AmeriFlux site
    end_lon = np.float32(np.array(data['end_lon'])[0])
    end_lat = np.float32(np.array(data['end_lat'])[0])
    end_agl = np.float32(np.array(data['end_agl'])[0])
    amf_lon = np.float32(np.array(data['ameriflux_lon'])[0])
    amf_lat = np.float32(np.array(data['ameriflux_lat'])[0])
    amf_agl = np.float32(np.array(data['ameriflux_agl'])[0])
    amf_time = np.float32(np.array(data['ameriflux_julian'])[0])
    amf_dist = 110*np.sqrt((amf_lon-ameriflux_lon)**2 + (amf_lat-ameriflux_lat)**2)
    # amfUSE  = amfDist .< maxDist
    if amf_dist < maxDist:
        source = 'ameriflux'
        bkg_conc = bkg_conc_ameriflux
        bkg_err = max(bkg_err, bkg_err_ameriflux)
    return bkg_conc, bkg_err, source

def parse_obs_info(file):
    """
        Parse the observation file details (timestamp, location etc.)

        Arguments:
            file: <str>
            
        returns:
            <list>
    """
    [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = file.replace('.nc', '').split("_")
    return [__, timestamp, receptor_lon, receptor_lat, receptor_agl]

def parse_ems_info(file):
    """
        Parse the emission file details (timestamp etc.)

        Arguments:
            file: <str>

        returns:
            <list>
    """
    trimmed_file = file.split("/")[-1].replace("_", "x")
    [_, year, month, day, hour] = trimmed_file.replace('.ncdf', '').split("x")
    return [_, year, month, day, hour]

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


def create_weights(x=1, r=0.5):
    n = back_hours+1
    a = x*(1-r)/(1-r**n)
    weight_list = []
    weight_list.append(a)
    term = a
    for idx in range(n-1):
        term = term*r
        weight_list.append(term)
    weight_list.reverse()
    return weight_list

def fill_obs_data(idx, foot_file, mode='integrated'):
    """
        Compute the observation data (H, Y, So_d etc.)

        Arguments:
            idx: <int>
            foot_file: <str>
            mode: <str>

        returns:
            
    """

    global H, Y, So_d, observation_dict, H_check, BKG

    [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = parse_obs_info(foot_file)
    receptor = [clon, clat, timestamp, receptor_lon, receptor_lat, idx]
    year = int(timestamp[0:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:])
    time_foot = datetime.datetime(year, month, day, hour)
    index = time_dict[time_foot]
    observation_dict[idx] = {
        "time":time_foot,
        "lon":float(receptor_lon),
        "lat":float(receptor_lat),
        "agl":float(receptor_agl)
    }


    foot_data = nc.Dataset(foot_file)
    bkg_conc_noaa = np.float32(np.array(foot_data['bkg_co2_NOAA'])[0])
    bkg_conc_nasa = np.float32(np.array(foot_data['bkg_co2_NASA'])[0])
    bkg_conc_ameriflux = np.float32(np.array(foot_data['ameriflux_co2'])[0])
    bkg_err_noaa = np.float32(np.array(foot_data['bkg_err_NOAA'])[0])
    bkg_err_nasa = np.float32(np.array(foot_data['bkg_err_NASA'])[0])
    bkg_err_ameriflux = np.float32(np.array(foot_data['ameriflux_err'])[0])
    bkg_conc, bkg_error, source = get_background_error(foot_data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux)
    Y[idx, 0] = np.float32(np.array(foot_data['co2'])[0]) - bkg_conc
    BKG[idx, 0] = bkg_conc
    # if Y[idx, 0] < 0:
    #     print(foot_file, np.float32(np.array(foot_data['co2'])[0]), bkg_conc, source)
    obs_error = np.float32(np.array(foot_data['co2_err'])[0])
    mod_error = model_error[time_foot.hour]
    So_d[idx, 0] = obs_error**2+bkg_error**2+mod_error**2
    if not emulator:
        if mode == 'integrated':
            foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            H[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
        elif mode == 'integrated_average':
            foot = np.float32(np.average(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            # To verify the code (code testing): foot_check and H_check
            foot_check = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot_check)

            # resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot)
        elif mode == 'integrated_decayed':
            # declare the weights
            weight_list = create_weights()
            foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200] #compressed footprints
            H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
            # import pdb; pdb.set_trace()
        elif mode == 'resolved':
            foot = np.float32(np.array(foot_data['foot']))[:, clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot[jdx, :, :])
    foot_data.close()
    return receptor

def readFootEmulator(receptor, emulator_run=emulator_run):
    [clon, clat, timestamp, receptor_lon, receptor_lat, idx] = receptor
    year = int(timestamp[0:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:])
    time_foot = datetime.datetime(year, month, day, hour)
    index = time_dict[time_foot]

    global H, H_check

    if emulator_run:
        pass
    else:
        file = f"{emulator_file_path}emulator_{timestamp}_{receptor_lon}_{receptor_lat}.nc"
        foot_data = nc.Dataset(file)
        foot = np.float32(np.array(foot_data['foot']))
        foot_data.close()
        if mode == 'integrated':
            H[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
        elif mode == 'integrated_average':
            # Validation (not used in the modeling)
            foot_step_avg = foot/back_hours
            H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

            # Used in modeling
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot_step_avg)
            pass
        elif mode == 'integrated_decayed':
            weight_list = create_weights()
            # Validation (not used in the modeling)
            H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

            # Used in the modeling
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
            pass
        else:
            raise Exception(f"Expected mode (in config.py) to be integrated, integrated_average, or integrated_decayed. Found mode to be {mode} instead.")

def runFootEmulator(receptor_batch, index_list):
    global H, H_check
    model = ftnet.footnet_model()
    model.load_weights(emulator_model_file)
    footprints = model.get_footprint(receptor_batch)
    for jdx, receptor in enumerate(receptor_batch):
        foot = footprints[jdx].footprint
        [clon, clat, timestamp, receptor_lon, receptor_lat] = receptor
        idx = index_list[jdx]
        year = int(timestamp[0:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = int(timestamp[8:])
        time_foot = datetime.datetime(year, month, day, hour)
        index = time_dict[time_foot]

        if emulator_run and emulator:
            if mode == 'integrated':
                H[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_average':
                # Validation (not used in the modeling)
                foot_step_avg = foot/back_hours
                H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

                # Used in modeling
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot_step_avg)

            elif mode == 'integrated_decayed':
                weight_list = create_weights()
                # Validation (not used in the modeling)
                H_check[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

                # Used in the modeling
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])

            else:
                raise Exception(f"Expected mode (in config.py) to be integrated, integrated_average, or integrated_decayed. Found mode to be {mode} instead.")


def fill_obs_validation(idx, foot_file, mode='integrated'):

    global H_valid, Y_valid, validation_dict, BKG_valid

    [__, timestamp, receptor_lon, receptor_lat, receptor_agl] = parse_obs_info(foot_file)
    receptor = [clon, clat, timestamp, receptor_lon, receptor_lat, idx]
    year = int(timestamp[0:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:])
    time_foot = datetime.datetime(year, month, day, hour)
    index = time_dict[time_foot]
    validation_dict[idx] = {
        "time":time_foot,
        "lon":float(receptor_lon),
        "lat":float(receptor_lat),
        "agl":float(receptor_agl)
    }

    foot_data = nc.Dataset(foot_file)
    bkg_conc_noaa = np.float32(np.array(foot_data['bkg_co2_NOAA'])[0])
    bkg_conc_nasa = np.float32(np.array(foot_data['bkg_co2_NASA'])[0])
    bkg_conc_ameriflux = np.float32(np.array(foot_data['ameriflux_co2'])[0])
    bkg_err_noaa = np.float32(np.array(foot_data['bkg_err_NOAA'])[0])
    bkg_err_nasa = np.float32(np.array(foot_data['bkg_err_NASA'])[0])
    bkg_err_ameriflux = np.float32(np.array(foot_data['ameriflux_err'])[0])
    bkg_conc, bkg_error, source = get_background_error(foot_data, bkg_conc_noaa, bkg_conc_nasa, bkg_conc_ameriflux, bkg_err_noaa, bkg_err_nasa, bkg_err_ameriflux)
    Y_valid[idx, 0] = np.float32(np.array(foot_data['co2'])[0]) - bkg_conc
    BKG_valid[idx, 0] = bkg_conc

    # H_valid
    if emulator:
        if not emulator_run:
            file = f"{emulator_file_path}emulator_{timestamp}_{receptor_lon}_{receptor_lat}.nc"
            foot_data = nc.Dataset(file)
            foot = np.float32(np.array(foot_data['foot']))
            if mode == 'integrated':
                H_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            elif mode == 'integrated_average':
                # Validation (not used in the modeling)
                foot_step_avg = foot/back_hours
                H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

                # Used in modeling
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot_step_avg)
                pass
            elif mode == 'integrated_decayed':
                weight_list = create_weights()
                # Validation (not used in the modeling)
                H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)

                # Used in the modeling
                resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
                for jdx, time_hour in enumerate(resolved_time_list):
                    m_index = time_dict[time_hour]
                    H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
            pass
    else:
        if mode == 'integrated':
            foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            H_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
        elif mode == 'integrated_average':
            foot = np.float32(np.average(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            # To verify the code (code testing): foot_check and H_check
            foot_check = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot_check)

            # resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot)
        elif mode == 'integrated_decayed':
            # declare the weights
            weight_list = create_weights()
            foot = np.float32(np.nansum(np.array(foot_data['foot']), axis=0))[clat_index-200:clat_index+200, clon_index-200:clon_index+200] #compressed footprints
            H_check_valid[idx, index*m:(index+1)*m] = flatten_2d_column(foot)
            resolved_time_list = date_range[date_range<=time_foot][-(back_hours+1):]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot*weight_list[jdx])
            # import pdb; pdb.set_trace()
        elif mode == 'resolved':
            foot = np.float32(np.array(foot_data['foot']))[:, clat_index-200:clat_index+200, clon_index-200:clon_index+200]
            resolved_time_list = date_range[date_range<=time_foot][-foot.shape[0]:]
            for jdx, time_hour in enumerate(resolved_time_list):
                m_index = time_dict[time_hour]
                H_valid[idx, m_index*m:(m_index+1)*m] = flatten_2d_column(foot[jdx, :, :])


def fill_obs_parallel():
    """
        Parallel implementation of the fill_obs_data function

        Arguments:

        returns:
            
    """
    print("Reading observation data ....")
    OUTPUT = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(fill_obs_data)(idx, foot_file, mode=mode) for idx, foot_file in tqdm(enumerate(domain_foot_files)))
    OUTPUT_valid = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(fill_obs_validation)(idx, foot_file, mode=mode) for idx, foot_file in tqdm(enumerate(domain_foot_files_valid)))
    if emulator:
        if emulator_run:
            print("Running emulator to generate footprints ....")
            receptor_batch = []
            index_list = []
            for receptor in OUTPUT:
                receptor_batch.append(receptor[:-1])  # excluding idx
                index_list.append(idx)
                runFootEmulator(receptor_batch, index_list)

        else:
            print("Reading footprints from stored emulator footprints .....")
            # print(OUTPUT)
            emulator_OUTPUT = Parallel(n_jobs=1, verbose=0, backend='threading')(delayed(readFootEmulator)(receptor, emulator_run=emulator_run) for receptor in tqdm(OUTPUT))
            # emulator_OUTPUT_valid = Parallel(n_jobs=-1, verbose=0, backend='threading')(delayed(readFootEmulator)(receptor, emulator_run=emulator_run) for receptor in tqdm(OUTPUT))
    return

def load_prior_emissions():
    """
        Read the prior emission data

        Arguments:

        returns:
            
    """
    global X
    print("Loading emission data ....")
    for idx, ems_file in tqdm(enumerate(emission_filtered_files)):
        [_, year, month, day, hour] = parse_ems_info(ems_file)
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        timestamp = datetime.datetime(year, month, day, hour)
        ems_data = np.float32(np.array(nc.Dataset(ems_file)['flx_total']))[clat_index-200:clat_index+200, clon_index-200:clon_index+200]
        ems_flattened = flatten_2d_column(ems_data)
        index = time_dict[timestamp]
        X[index*m:(index+1)*m, 0] = ems_flattened

def fill_observation_covariance_matrix():
    """
        Compute the observational covariance matrix

        Arguments:

        returns:
            
    """
    global So
    # So = np.zeros((So_d.shape[0], So_d.shape[0]), dtype=np.float32)
    print("Forming observation covariance matrix ....")
    nObs = So_d.shape[0]
    
    # Adding diagonal terms
    for i in range(nObs):
        So[i, i] = So_d[i, 0]

    # Adding off-diagonal terms
    for i in tqdm(range(nObs)):
        time_val_i = observation_dict[i]['time']
        coord_i = (observation_dict[i]['lat'], observation_dict[i]['lon'])
        for j in range(i+1, nObs):
            coord_j = (observation_dict[j]['lat'], observation_dict[j]['lon'])
            time_val = (observation_dict[j]['time']-time_val_i).seconds/3600
            dist_val = get_len(coord_i, coord_j)
            time_decay = np.exp(-abs(time_val)/tau_time)
            dist_decay = np.exp(-abs(dist_val)/tau_space)
            sig_val = time_decay*dist_decay*np.sqrt(So_d[i]*So_d[j])
            if time_decay*dist_decay > lowBound:
                So[i, j] = sig_val
                So[j, i] = sig_val
    So = csc_matrix(So)


fill_obs_parallel()
load_prior_emissions()
fill_observation_covariance_matrix()

