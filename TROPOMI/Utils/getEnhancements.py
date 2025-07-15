import numpy as np

def compute_Y_So_d(H_Y_dict, indices):
    Y = np.zeros((len(indices), 1), dtype=np.float32)
    So_d = np.zeros((len(indices), 1), np.float32)
    BKG = np.zeros((len(indices), 1), np.float32)
    for i, idx in enumerate(indices):
        # bkg, bkg_error = get_background_value(idx)
        bkg = H_Y_dict[idx]['bkg']
        bkg_error = H_Y_dict[idx]['bkg_error']
        mod_error = H_Y_dict[idx]['methane_mixing_ratio_precision']**2     # Need to change it to model error
        Y[i, 0] = H_Y_dict[idx]['methane_mixing_ratio_bias_corrected'] - bkg
        So_d[i, 0] = H_Y_dict[idx]['methane_mixing_ratio_precision']**2 + bkg_error**2 + mod_error**2
        BKG[i, 0] = H_Y_dict[idx]['bkg']
    return Y, So_d, BKG

def compute_Y_valid(H_Y_dict, indices):
    Y = np.zeros((len(indices), 1), dtype=np.float32)
    BKG = np.zeros((len(indices), 1), np.float32)
    for i, idx in enumerate(indices):
        bkg = H_Y_dict[idx]['bkg']
        bkg_error = H_Y_dict[idx]['bkg_error']
        mod_error = H_Y_dict[idx]['methane_mixing_ratio_precision']**2     # Need to change it to model error
        Y[i, 0] = H_Y_dict[idx]['methane_mixing_ratio_bias_corrected'] - bkg
        BKG[i, 0] = H_Y_dict[idx]['bkg']
    return Y, BKG
