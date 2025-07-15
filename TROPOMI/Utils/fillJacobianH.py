import numpy as np
# from config import *


def compute_H(H_Y_dict, indices, date_range, m, time_dict):
    H = np.zeros((len(indices), date_range.shape[0]*m))
    for i, idx in enumerate(indices):
        resolved_time_list = date_range[date_range<=H_Y_dict[idx]['time']][-4:]
        for jdx, time_day in enumerate(resolved_time_list):
            m_index = time_dict[time_day]
            H[i, m_index*m:(m_index+1)*m] = H_Y_dict[idx]['gp_foot'][jdx].reshape(-1, order='F')
    H = H*1000 # Conversion to ppb from ppm
    return H