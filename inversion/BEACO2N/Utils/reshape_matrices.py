# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:46:07
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 14:43:47

import numpy as np



def flatten_2d_column(foot):
    """
        Flatten the 2D array into a vector

        Arguments:
            foot: <2D array>
        returns:
            sub_foot: <1D array>
    """
    sub_foot = np.zeros((foot.shape[0]*foot.shape[1]), dtype=np.float32)
    for idx in range(foot.shape[1]):
        sub_foot[idx*foot.shape[0]:(idx+1)*foot.shape[0]] = foot[:, idx]
    return sub_foot


def make_grid_2d_column(X, nrow, ncol):
    """
        Convert vector into 2D array

        Arguments:
            X: <1D array>
            nrow: <int>
            ncol: <int>
        returns:
            X1: <2D array>
    """
    X1 = np.zeros((nrow, ncol), dtype=np.float32)
    for idx in range(ncol):
        X1[:, idx] = X[idx*nrow:(idx+1)*nrow, 0]
    return X1