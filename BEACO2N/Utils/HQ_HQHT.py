# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-08-28 19:28:02
# @Last Modified by:   nd349
# @Last Modified time: 2024-09-10 17:11:27


import time
import numpy as np
from scipy import sparse
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

# def HQ(H, D, E):
#     '''
#     Function to compute matrix multiplication of H and Prior error covariance matrix (HQ) 
#     where Q = DxE (kronecker product)
#     Input
#     H: 2 D numpy matrix or sparse matrix
#     D: 2 D numpy matrix or sparse matrix
#     E: 2 D sparse matrix

#     Output
#     HQ: 2 D numpy matrix
#     '''
#     n = H.shape[0]
#     p, q = D.shape
#     r, t = E.shape
#     HQ = np.zeros((n, q*t), dtype=np.float32)
#     counter = t
#     E = csc_matrix(E)
#     for i in tqdm(range(q)):
#         temp = (i+1)*t
#         HQsum = np.zeros((n, r), dtype=np.float32)
#         for j in range(p):
#             if D[j, i] != 0 and D[j, i] != 1:
#                 HQsum += H[:, (j)*r:(j+1)*r]*D[j, i]
#             elif D[j, i] == 1:
#                 HQsum += H[:, j*r:(j+1)*r]
#         # HQsum = coo_matrix(HQsum)
#         HQ[:, temp-counter:temp] = csc_matrix.dot(csc_matrix(HQsum), E).toarray()
#     return HQ

class computeHQ():
    def __init__(self, H, D, E, parallel):
        n = H.shape[0]
        p, q = D.shape
        r, t = E.shape
        self.n = H.shape[0]
        self.p, self.q = D.shape
        self.r, self.t = E.shape
        self.H = H
        self.D = D
        self.E = csc_matrix(E)
        self.HQ = np.zeros((self.n, self.q*self.t), dtype=np.float32)
        self.parallel = parallel
        self.counter = self.t
    
    
    def computeHQSerial(self):
        H = self.H
        D = self.D
        E = self.E
        n = H.shape[0]
        p, q = D.shape
        r, t = E.shape
        counter = t
        print("computing HQ sequentially ....")
        for i in tqdm(range(q)):
            temp = (i+1)*t
            HQsum = np.zeros((n, r), dtype=np.float32)
            for j in range(p):
                if D[j, i] != 0 and D[j, i] != 1:
                    HQsum += H[:, (j)*r:(j+1)*r]*D[j, i]
                elif D[j, i] == 1:
                    HQsum += H[:, j*r:(j+1)*r]
            self.HQ[:, temp-counter:temp] = csc_matrix.dot(csc_matrix(HQsum), E).toarray()
    
    def computeHQthread(self, i):
        D = self.D
        H = self.H
        E = self.E
        counter = self.t
        n = self.n
        r = self.r
        q = self.q
        t = self.t
        p = self.p
        temp = (i+1)*t
        HQsum = np.zeros((n, r), dtype=np.float32)
        for j in range(p):
            if D[j, i] != 0 and D[j, i] != 1:
                HQsum += H[:, (j)*r:(j+1)*r]*D[j, i]
            elif D[j, i] == 1:
                HQsum += H[:, j*r:(j+1)*r]
        # HQsum = coo_matrix(HQsum)
        self.HQ[:, temp-counter:temp] = csc_matrix.dot(csc_matrix(HQsum), E).toarray()

    def computeHQParallel(self):
        OUTPUT = Parallel(n_jobs=64, verbose=10, backend='threading')(delayed(self.computeHQthread)(i) for i in tqdm(range(self.q)))


def HQ(H, D, E, parallel=True):
    hq = computeHQ(H, D, E, parallel)
    if parallel:
        print("Parallel computation of HQ ....")
        hq.computeHQParallel()
        return hq.HQ
    else:
        print("Sequential computation of HQ ...")
        hq.computeHQSerial()
        return hq.HQ


def HQHT(HQ_INDIRECT, H, D, E):
    '''
    Function to compute HQHT where Q is kronecker product of D and E
    Input
    HQ: 2 D numpy matrix
    H: 2 D numpy matrix or sparse matrix
    D: 2 D numpy matrix or sparse matrix
    E: 2 D sparse matrix (csc matrix)

    Output
    HQHT: 2 D numpy matrix
    '''

    n = H.shape[0]
    p, q = D.shape
    r, t = E.shape
    HQHT = np.zeros((n, n), dtype = np.float32)
    counter = t
    for i in tqdm(range(q)):
        # import pdb; pdb.set_trace()
        # print(i)
        temp = (i+1)*t
        HQHT += np.dot(HQ_INDIRECT[:, temp-counter:temp], H[:, (i)*r:(i+1)*r].T)
        # HQHT += csc_matrix.dot(csc_matrix(HQ[:, temp-counter:temp]), csc_matrix(H[:, (i)*r:(i+1)*r].T)).toarray()
    return HQHT

def computeQHTeta(H, D, E, eta):
    H_prime = csc_matrix(np.dot(H.T, eta))
    m1 = E.shape[0]
    QHeta = []
    for j in tqdm(range(D.shape[0])):
        A1 = csc_matrix(np.zeros((E.shape[0], 1)))
        for i in range(D.shape[0]):
            if D[j, i] !=0 and D[j, i]!=1:
                A1 += H_prime[m1*i:(i+1)*m1]*D[j, i]
            elif D[j, i] == 1:
                A1 += H_prime[m1*i:(i+1)*m1]
        temp = csc_matrix.dot(E, A1)
        QHeta += list(temp.toarray())
    QHeta = np.array(QHeta)
    return QHeta

def form_matrices():
    p = 100
    q = 1000
    r = 100
    t = 1000
    n = 5000
    m1 = p*r
    print("Random matrices are being initialized ....")
    D = csc_matrix(sparse.random(p, q,density=0.5,)).toarray()
    print("D is initialized")
    E = csc_matrix(sparse.random(r, t,density=0.5,))
    print("E is initialized")
    H = csc_matrix(sparse.random(n, p*r,density=0.5,)).toarray()
    print("H is initialized")
    print(H.shape, D.shape, E.shape)
    return H, D, E

if __name__ == '__main__':
    
    H, D, E = form_matrices()
    
    # start = time.time()
    # Q = np.kron(D, E.toarray())
    # print("Matrix multiplication starts ...")
    # HQ_DIRECT = np.float32(np.dot(H, Q))
    # print(HQ_DIRECT)
    # print(HQ_DIRECT.shape)
    # print("Time direct HQ:", time.time()-start)

    start = time.time()
    HQ_INDIRECT = HQ(H, D, E, parallel=True)
    print(HQ_INDIRECT)
    print(HQ_INDIRECT.shape)
    print("Time indirect HQ:", time.time()-start)


    start = time.time()
    HQHT_DIRECT = np.float32(np.dot(H, np.dot(Q, H.T)))
    print(HQHT_DIRECT)
    print(HQHT_DIRECT.shape)
    print("Time direct HQHT:", time.time()-start)

    start = time.time()
    HQHT_INDIRECT = HQHT(HQ_INDIRECT, H, D, E) 
    print(HQHT_INDIRECT)
    print(HQHT_INDIRECT.shape)
    print("Time indirect HQHT:", time.time()-start)

