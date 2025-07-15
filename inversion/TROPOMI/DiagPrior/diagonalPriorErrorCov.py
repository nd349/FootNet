import numpy as np
from config import *

from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

def compute_diagonal_prior_error_covariance(Xa):
    B_diag = np.square(ems_uncert*Xa.toarray())
    B_row = np.array([i for i in range(Xa.shape[0])])
    # Sa_d = np.square(self.ems_uncert*self.X_pri_array)
    # Sa_d[Sa_d<self.minUncert**2]=self.minUncert**2
    B = csc_matrix((B_diag[:, 0], (B_row, B_row)), shape=(Xa.shape[0], Xa.shape[0]), dtype=np.float32)
    return B