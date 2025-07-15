# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2024-06-15 14:03:29
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2024-06-15 20:48:28

import time
import numpy as np
from DiagPrior.diagonalPriorErrorCov import compute_diagonal_prior_error_covariance
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv

from config import *


class diagPriorInversion():
    def __init__(self, H, Xp, Y, R, H_Y_dict):
        self.H = csc_matrix(H)
        self.Xp = csc_matrix(Xp)
        print("Dim of Xp:", self.Xp.shape)
        self.Y = csc_matrix(Y)
        self.R = csc_matrix(R)
        self.H_Y_dict = H_Y_dict
        self.B = compute_diagonal_prior_error_covariance(self.Xp)
        self.buffer_days = buffer_days


    def invert(self):
        start = time.time()
        # import pdb; pdb.set_trace()
        mismatch = self.Y - csc_matrix.dot(self.H, self.Xp)
        HB = csc_matrix.dot(self.H, self.B)
        HBHT = csc_matrix.dot(HB, self.H.T)
        G = HBHT + self.R
        X_diff = csc_matrix.dot(HB.T, csc_matrix.dot(inv(G), mismatch))
        X_diff = X_diff.reshape(-1, 1)
        print("Dims of X_diff", X_diff.shape, type(X_diff))
        self.X_hat = self.Xp + X_diff
        print(f"Time taken for inversion: {time.time()-start} seconds")
        return self.remove_padding(self.X_hat)


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
        X_week = X_hat.toarray()[buffer_days*m:(X_hat.shape[0]-buffer_days*m)]
        X_post = np.zeros((int(X_week.shape[0]/m), nrow, ncol))
        for idx in range(int(X_week.shape[0]/m)):
            X_post[idx, :, :] = X_week[idx*m:(idx+1)*m].reshape(nrow, ncol, order='F')
        return X_post



    



