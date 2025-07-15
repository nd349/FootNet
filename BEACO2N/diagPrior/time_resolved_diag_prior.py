# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-23 12:48:32
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-10-11 14:59:18


import numpy as np
from config import *
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import inv
import torch, time

class InversionDiagPrior():

    def __init__(self, H, X, Y, So_d):
        """
            Initializing the class 

            Arguments:
                H: <1D array>
                X: <1D array>
                Y: <1D array>
            returns:
        """
        self.mode = mode
        self.diag_prior = diag_prior
        self.sparse = sparse
        self.ems_uncert = ems_uncert
        self.minUncert = minUncert
        self.device = device
        if not self.diag_prior:
            raise Exception(f"diag_prior is expected to be True but found this instead: {diag_prior}")
        if self.sparse:
            self.H = csc_matrix(H)
            self.X_pri = csc_matrix(X)
            self.X_pri_array = X
            self.Y = csc_matrix(Y)
        else:
            self.H = H
            self.X_pri = X
            self.Y = Y
        
        self.So = self.form_So(So_d)
        self.Sa = self.form_Sa()


    def form_So(self, So_d):
        """
            Compute the observational covariance matrix

            Arguments:
                So_d: <2D array>
            returns:
                Sa
        """
        So_row = np.array([i for i in range(self.Y.shape[0])])
        So = csc_matrix((So_d[:, 0], (So_row, So_row)), shape=(self.Y.shape[0], self.Y.shape[0]), dtype=np.float32)
        # So = csc_matrix(So, dtype=np.float32)
        return So

    def form_Sa(self):
        """
        Compute the prior covariance matrix

        Arguments:
    
        returns:
            Sa: <2D array>
        """
        # import pdb; pdb.set_trace()
        Sa_row = np.array([i for i in range(self.X_pri.shape[0])])
        Sa_d = np.square(self.ems_uncert*self.X_pri_array)
        Sa_d[Sa_d<self.minUncert**2]=self.minUncert**2
        Sa = csc_matrix((Sa_d[:, 0], (Sa_row, Sa_row)), shape=(self.X_pri.shape[0], self.X_pri.shape[0]), dtype=np.float32)
        # Sa = csc_matrix(Sa, dtype=np.float32)
        return Sa

    def transfer_to_torch(self, a):
        """
        transfer to pytorch

        Arguments:
            a: <2D sparse matrix>

        returns:
            a: <tensor pytorch>
        """
        values = a.data
        indices = np.vstack((a.row, a.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = a.shape

        a = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)
        return a

    def invert(self):
        """
        Invert for posterior solution

        Arguments:
            
        returns:
            posterior solution
        """
        print("Inversion is starting .....")
        print(f"Size of H: {self.H.shape}")
        print(f"Size of X prior: {self.X_pri.shape}")
        print(f"Size of Y: {self.Y.shape}")
        start = time.time()
        if self.device == 'cpu':
            mismatch = self.Y - csc_matrix.dot(self.H, self.X_pri)
            KSa = csc_matrix.dot(self.H, self.Sa)
            G = csc_matrix.dot(KSa, self.H.T) + self.So
            mismatch1 = csc_matrix.dot(inv(G), mismatch)
            X_dif = csc_matrix.dot(KSa.T, mismatch1)
            X_hat = self.X_pri + X_dif
            self.X_hat = X_hat
            print(f"Time taken for inversion: {time.time()-start}")
            return X_hat

        elif self.device == 'cuda' or self.device == 'gpu':
            # Transfer to torch
            H = coo_matrix(self.H)
            Y = coo_matrix(self.Y)
            X_pri = coo_matrix(self.X_pri)
            So = coo_matrix(self.So)
            Sa = coo_matrix(self.Sa)

            print(f"Transfering to torch ({self.device}) ... ")
            H_d = self.transfer_to_torch(H)
            Y_d = self.transfer_to_torch(Y)
            X_pri_d = self.transfer_to_torch(X_pri)
            So_device = self.transfer_to_torch(So)
            Sa_device = self.transfer_to_torch(Sa)
            

            # Inversion
            # import pdb; pdb.set_trace()
            mismatch_d = Y_d - torch.sparse.mm(H_d, X_pri_d)
            KSa_d = torch.sparse.mm(H_d, Sa_device)
            G_d = torch.sparse.mm(KSa_d, torch.t(H_d)) + So_device
            mismatch1_d = torch.mm(torch.inverse(G_d.to_dense()), mismatch_d.to_dense()).to_sparse()
            X_dif_d = torch.sparse.mm(torch.t(KSa_d), mismatch1_d)
            X_hat_d = X_pri_d + X_dif_d
            X_hat_host_sparse = csc_matrix(X_hat_d.to_dense().cpu().detach().numpy())
            self.X_hat = X_hat_host_sparse
            print(f"Time taken for inversion: {time.time()-start}")
            return X_hat_host_sparse

        else:
            raise Exception(f"Expected device from [cpu, cuda] but found this instead: {self.device}")







