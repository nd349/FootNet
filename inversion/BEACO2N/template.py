# -*- coding: utf-8 -*-
# @Author: nikhildadheech
# @Date:   2022-07-25 14:58:38
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2023-12-02 18:36:50

# from diagPrior.time_resolved_diag_prior import InversionDiagPrior

from Utils.getData import *
from config import *
from fullCovariance.InversionFullCovariance import InversionFullPrior
import pickle

# if cross_validation:
# 	print(Y_valid)

invert = InversionFullPrior(H, X, Y, So, observation_dict, BKG)
X_hat = invert.invert()
print("Size of the posterior solution:", X_hat.shape)
invert.save_solution()

if cross_validation:
	# print(Y_valid)
	invert.save_concentrations(H_valid, Y_valid, validation_dict, BKG_valid)
else:
	invert.save_concentrations()

# with open(output_posterior_file, "wb") as write_file:
# 	pickle.dump(X_hat, write_file)