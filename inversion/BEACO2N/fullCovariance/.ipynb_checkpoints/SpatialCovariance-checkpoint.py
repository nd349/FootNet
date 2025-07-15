# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-25 12:14:36
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 15:18:48


import pickle
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix


Sa_xy_file = "data/Sa_xy_corrcoef_emulator.pkl"


def load_Sa_xy(file):
	"""
        Load spatial correlation matrix

        Arguments:
            file: <str>
        returns:
            Sa_xy: <csc matrix>
    """
	with open(file, 'rb') as open_file:
		Sa_xy = pickle.load(open_file)
	Sa_xy = csc_matrix(Sa_xy)
	return Sa_xy

# Sa_xy = load_Sa_xy(Sa_xy_file)