import warnings
warnings.filterwarnings("ignore")

import torch, time
import datetime
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random, os
from FootNet.unet_model import UNet


class FootNet():
    def __init__(self, device, n_channels, n_classes, emulator_model_path, epsilon):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.emulator_model_path = emulator_model_path
        self.device = device
        self.epsilon = epsilon
        self.model = UNet(n_channels=n_channels, n_classes=n_classes).to(self.device)
        self.load_model(self.model, emulator_model_path)

    def load_model(self, model, filename):
        # print(f"...Loading {filename}")
        # checkpoint = torch.load(filename)
        # model.load_state_dict(checkpoint['model'])
        print(f"...Loading {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['MODEL_STATE'])

    def run_inference(self, batch_DG):
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(batch_DG):
                inputs, reference, foot_hours = data
                inputs = inputs.to(self.device, dtype=torch.float)
                prediction = self.model(inputs)
                if idx == 0:
                    prediction_valid = prediction.cpu()
                    reference_valid = reference
                    foot_hours_valid = foot_hours
                    
                else:
                    prediction_valid = torch.cat([prediction_valid, prediction.cpu()], axis=0)
                    reference_valid[0] = torch.cat([reference_valid[0], reference[0]], axis=0)
                    reference_valid[1] += reference[1]
                    reference_valid[2] = torch.cat([reference_valid[2], reference[2]], axis=0)
                    reference_valid[3] = torch.cat([reference_valid[3], reference[3]], axis=0)
                    foot_hours_valid = torch.cat([foot_hours_valid, foot_hours], axis=0)
                    
        foots = self.transform(prediction_valid)[:, 0, :, :].cpu().detach().numpy()
        reference_indices = reference_valid[0].cpu().detach().numpy()
        reference_timestamps = reference_valid[1]
        reference_rlons = reference_valid[2].cpu().detach().numpy()
        reference_rlats = reference_valid[3].cpu().detach().numpy()
        foot_hours_valid = foot_hours_valid.cpu().detach().numpy()
        return foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, foot_hours_valid

    def transform(self, pred):
        epsilon = self.epsilon
        pred = pred/1000
        pred = pred + np.log(epsilon)
        pred = np.exp(pred) - epsilon
        pred[np.where(pred<0)] = 0
        return pred

    def create_weights(self, back_hours, x=1, r=0.5):
        n = back_hours
        a = x*(1-r)/(1-r**n)
        weight_list = []
        weight_list.append(a)
        term = a
        for idx in range(n-1):
            term = term*r
            weight_list.append(term)
        weight_list.reverse()
        return weight_list

    def compute_daily_resolved_footprint(self, foots, foot_hours, data, nrow, ncol, transport_max_days, transport_max_backhours):
        avg_foot_hours = min(round(np.average(foot_hours)), transport_max_backhours)
        weights = self.create_weights(avg_foot_hours)
        foot_avg = np.average(foots, axis=0)
        gp_foot = np.zeros((transport_max_backhours, nrow, ncol))
        for kdx in range(1, len(weights)+1):
            gp_foot[-kdx, :, :] = weights[-kdx]*foot_avg
            

        st_end = data['time'][0] + datetime.timedelta(hours=23-data['time'][0].hour)
        et_end = st_end - datetime.timedelta(hours=transport_max_days*24 - 1)
        foot_date_range = pd.date_range(start=et_end, end=st_end, freq='1h')
        foot_tstamp_indices = [idx for idx in range(foot_date_range.get_loc(data['time'][0])-(transport_max_backhours-1), foot_date_range.get_loc(data['time'][0])+1)]

        foot_H = np.zeros((foot_date_range.shape[0], nrow, ncol))
        for idx, index in enumerate(foot_tstamp_indices):
            foot_H[index, :, :] = gp_foot[idx, :, :]

        h3 = np.average(foot_H[-24:], axis=0)
        h2 = np.average(foot_H[-48:-24], axis=0)
        h1 = np.average(foot_H[-72:-48], axis=0)
        h0 = np.average(foot_H[-96:-72], axis=0)
        h_total = np.zeros((transport_max_days, nrow, ncol))
        h_total[0, :, :] = h0
        h_total[1, :, :] = h1
        h_total[2, :, :] = h2
        h_total[3, :, :] = h3
        return h_total, foot_date_range, avg_foot_hours
        