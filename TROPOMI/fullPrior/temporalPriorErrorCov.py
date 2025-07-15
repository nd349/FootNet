import numpy as np
from tqdm.auto import tqdm
# from config import *


def compute_temporal_prior_error_covariance(Xa, tau_week, tau_day, ems_uncert, m):
	nEms = int(Xa.shape[0]/m)
	nG = m

	Sa_t = np.zeros((nEms, nEms))

	for i in tqdm(range(nEms)):
	    sigmai = np.average(Xa[i*m:(i+1)*m])*ems_uncert
	    for j in range(i, nEms):
	        sigmaj = np.average(Xa[j*m:(j+1)*m])*ems_uncert
	        days_apart = j - i
	        weeks_apart = days_apart/7
	        temp_days = np.exp(-abs(days_apart)/tau_day)
	        temp_weeks = np.exp(-abs(weeks_apart)/tau_week)

	        # sig_val = np.sqrt(sigmai*sigmaj)*temp_days*temp_weeks
	        sig_val = np.sqrt(sigmai*sigmaj)*temp_days
	        Sa_t[i, j] = sig_val
	        Sa_t[j, i] = sig_val

	return Sa_t


def plot_temporal_prior_error_covariance(Sa_t, idx):

	nEms = int(Xa.shape[0]/m)
	index = [i for i in range(nEms)]
	h = plt.pcolor(index, index, Sa_t)
	plt.colorbar(h)
	plt.title("Temporal prior error covariance matrix (D)")
	plt.show()

	value = Sa_t[idx, :]
	plt.plot(value)
	plt.title(f"Temporal covariance matrix slice for timestep: {idx}")
