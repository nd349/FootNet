import glob
import warnings
import argparse
warnings.filterwarnings("ignore")

import torch, time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random, os, socket
import netCDF4 as nc
from torch.utils.tensorboard import SummaryWriter
from os import listdir
# from torchmetrics import R2Score, F1Score
import xarray as xr
from sklearn.model_selection import train_test_split

# from unet_model import UNet
from unetpp_model import NestedUNet

host = os.uname()[1]
print(host)
def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std

def get_distance(timestamp, rlon, rlat, lat, lon):
    # print(file)
    # print(file.split("/")[-1].split("_")[3:6])
    # timestamp, rlon, rlat = file.split("/")[-1].split("_")[3:6]
    # rlon = float(rlon)
    # rlat = float(rlat)
    # # print(timestamp, rlon, rlat)
    # data = nc.Dataset(file)
    # lat = np.array(data['lat'])
    # lon = np.array(data['lon'])
    
    rlat_index = np.unravel_index((np.abs(lat- rlat)).argmin(), lat.shape)
    rlon_index = np.unravel_index((np.abs(lon- rlon)).argmin(), lon.shape)
    # print(rlat_index, rlon_index, rlon, rlat)
    # data.close()
    
    lon, lat = np.meshgrid(lon, lat)
    
    lon = lon*np.pi/180
    lat = lat*np.pi/180
    rlon = rlon*np.pi/180
    rlat = rlat*np.pi/180

    a = np.sin((lat-rlat)/2)**2 + np.cos(lat)*np.cos(rlat)*(np.sin((lon-rlon)/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6371e3*c/1000
    return d

def transform_func_24h(_xx, _6xx, _12xx, _18xx, _24xx, _yy, comb_plume, gp_first, transform=''):
    '''
    xx: (400, 400, 14)
    yy: (400, 400)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    #          'U10M', 'V10M', 'PBLH', 'PRSS', 'UWND9_850hPa', 'VWND9_850hPa', 'UWND17_500hPa', 'VWND17_500hPa', 'PRES9_850hPa', 'TEMP9_850hPa'
    # SCALERS = [1e1,     1e1,    1e-3,   1e-3]
    SCALERS = [1e1, 1e1, 1e-3, 1e-3, 1, 1, 1, 1, 1e-3, 1e-2]
    # SCALERS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0, 0, 0, 0]
    
    original_yy = _yy.copy()
    if transform=='log_square':
        _yy[np.where(_yy == 0)] = np.nan
        _yy = (np.log(_yy) + 30)**2
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='log':
        _yy[np.where(_yy <= 1e-8)] = np.nan
        _yy = np.log(_yy) + 30
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='multiply':
        _yy = _yy*1e5
    elif transform=="log-epsilon_xstilt":
        epsilon = 1e-4
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
    elif transform=="log-epsilon-threshold":
        epsilon = 1e-3
        _yy[np.where(_yy <= 1e-6)] = 0
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
    else:
        print("No Transformation....")
        
    _yy = _yy[:, :, np.newaxis]  # 400, 400, 1
    
    for i in range(_xx.shape[2]):
        # _xx[:, :, i] = zstandard(_xx[:, :, i])
        _xx[:, :, i] = _xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    _xx = np.delete(_xx, [8], axis=-1) # 400, 400, X
    
    for i in range(_6xx.shape[2]):
        # _6xx[:, :, i] = zstandard(_6xx[:, :, i])
        _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    _6xx = np.delete(_6xx, [8], axis=-1) # 400, 400, X
    
    
    for i in range(_12xx.shape[2]):
        # _12xx[:, :, i] = zstandard(_12xx[:, :, i])
        _12xx[:, :, i] = _12xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    _12xx = np.delete(_12xx, [8], axis=-1) # 400, 400, X
    
    for i in range(_18xx.shape[2]):
        # _xx[:, :, i] = zstandard(_xx[:, :, i])
        _18xx[:, :, i] = _18xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    _18xx = np.delete(_18xx, [8], axis=-1) # 400, 400, X
    
    for i in range(_24xx.shape[2]):
        # _xx[:, :, i] = zstandard(_xx[:, :, i])
        _24xx[:, :, i] = _24xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    _24xx = np.delete(_24xx, [8], axis=-1) # 400, 400, X
    
    comb_plume = np.array(comb_plume)[:, :, np.newaxis]
    comb_plume[np.where(comb_plume>=0.08)] = 1
    comb_plume[np.where(comb_plume<0.08)] = 0

    gp_first = zstandard(gp_first)
    gp_first = gp_first[:, :, np.newaxis]
    return np.concatenate([gp_first, _xx, _6xx, _12xx, _18xx, _24xx, comb_plume, _yy], axis=-1), original_yy

def transform_func_12h(_xx, _6xx, _12xx, _yy, comb_plume, gp_first, transform=''):
    '''
    xx: (400, 400, 14)
    yy: (400, 400)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    #          'U10M', 'V10M', 'PBLH', 'PRSS', 'UWND9_850hPa', 'VWND9_850hPa', 'UWND17_500hPa', 'VWND17_500hPa', 'PRES9_850hPa', 'TEMP9_850hPa'
    # SCALERS = [1e1,     1e1,    1e-3,   1e-3]
    SCALERS = [1e1, 1e1, 1e-3, 1e-3, 1, 1, 1, 1, 1e-3, 1e-2]
    # SCALERS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0, 0, 0, 0]
    
    for i in range(_xx.shape[2]):
        # _xx[:, :, i] = zstandard(_xx[:, :, i])
        _xx[:, :, i] = _xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    original_yy = _yy.copy()
    if transform=='log_square':
        _yy[np.where(_yy == 0)] = np.nan
        _yy = (np.log(_yy) + 30)**2
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='log':
        _yy[np.where(_yy <= 1e-8)] = np.nan
        _yy = np.log(_yy) + 30
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='multiply':
        _yy = _yy*1e5
    elif transform=="log-epsilon_xstilt":
        epsilon = 1e-4
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
    elif transform=="log-epsilon-threshold":
        epsilon = 1e-3
        _yy[np.where(_yy <= 1e-6)] = 0
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
    else:
        print("No Transformation....")
        
    _yy = _yy[:, :, np.newaxis]  # 400, 400, 1
    
    
    for i in range(_6xx.shape[2]):
        # _6xx[:, :, i] = zstandard(_6xx[:, :, i])
        _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    
    for i in range(_12xx.shape[2]):
        # _12xx[:, :, i] = zstandard(_12xx[:, :, i])
        _12xx[:, :, i] = _12xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    
    comb_plume = np.array(comb_plume)[:, :, np.newaxis]
    comb_plume[np.where(comb_plume>=0.08)] = 1
    comb_plume[np.where(comb_plume<0.08)] = 0

    gp_first = zstandard(gp_first)
    gp_first = gp_first[:, :, np.newaxis]
    return np.concatenate([gp_first, _xx, _6xx, _12xx, comb_plume, _yy], axis=-1), original_yy

class FootDataset(Dataset):
    def __init__(self, files, transform=None, limit=None, extension='.nc', backhours=12):
        self.data_files = files
        self.limit = limit
        self.transform = transform
        self.extension = extension
        self.backhours = backhours
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file = self.data_files[idx]
        try:
            # print(file)
            timestamp, rlon, rlat = file.split("/")[-1].replace(".nc", "").split("_")[3:6]
            rlon = float(rlon)
            rlat = float(rlat)
            
            # print('random', file)
            # file = "/home/h6/nd349/xstilt_CONUS_data/" + file.split("/")[-1]
            # file = f"{data_path}{file.split('/')[-1]}"
            
            data = self.load_data(file)
            
            lat = data['lat'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200]
            lon = data['lon'][int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
            
            dist = get_distance(timestamp, rlon, rlat, lat, lon)
            combined_gp = data['combined_gaussian_plume'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
            gp_first = data['gaussian_plume'][:, :, 0][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
            
            label = data['obs'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]    
            
            _pred = data['_xpred'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
            _6hpred = data['_x6hpred'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
            _12hpred = data['_x12hpred'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
    
            if self.backhours == 12:
                tempxy, original_yy = transform_func_12h(_pred, _6hpred, _12hpred, label, combined_gp, gp_first, transform=self.transform)
            elif self.backhours == 24:
                _18hpred = data['_x18hpred'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
                _24hpred = data['_x24hpred'][int(np.array(data['clat_shift_index'])[0])-200:int(np.array(data['clat_shift_index'])[0])+200, int(np.array(data['clon_shift_index'])[0])-200:int(np.array(data['clon_shift_index'])[0])+200]
                tempxy, original_yy = transform_func_24h(_pred, _6hpred, _12hpred, _18hpred, _24hpred, label, combined_gp, gp_first, transform=self.transform)
            
            
            tempx = tempxy[:, :, :-1] # Separating label from input
            tempxx = np.zeros((tempx.shape[2], tempx.shape[0], tempx.shape[1]))
            
            tempxx = np.concatenate([tempxx, dist[np.newaxis, :, :], np.exp(0.01*dist)[np.newaxis, :, :]], axis=0)
            
            for idx in range(tempx.shape[2]):
                tempxx[idx, :, :] = tempx[:, :, idx]
                
            tempy = tempxy[:, :, -1] # Getting label
            tempy = np.array(tempy[np.newaxis, :, :])
            
            original_yy = np.array(original_yy[np.newaxis, :, :])
                
            if self.extension == ".nc" or self.extension == ".ncdf":
                data.close()
            return tempxx, tempy, original_yy, file
        except Exception as e:
            print(e, file)
    
    def load_data(self, file):
        if self.extension == '.npz':
            return np.load(file)
        elif self.extension == ".nc" or self.extension == ".ncdf":
            # try:
            return nc.Dataset(file)
            # except Exception as e:
            #     print(e, file)

def getR2(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        snapshot_path: str,
        train_DG: DataLoader,
        valid_DG = None,
        location = None,
        experiment = None,
        early_stop_thresh = None,
        args = None,
        customized_loss = None
        
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_DG = train_DG
        self.valid_DG = valid_DG
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.epochs_run = 1
        self.best_valid_loss = np.inf
        self.best_valid_epoch = 0
        self.early_stop_thresh = early_stop_thresh
        self.stat_list = []
        self.snapshot_path = snapshot_path
        self.location = location
        self.experiment = experiment
        self.args = args
        self.customized_loss = customized_loss
        print(self.customized_loss)
        
        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot: {self.snapshot_path}")
            self._load_snapshot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc, weights_only=False)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.best_valid_loss = snapshot["valid_loss"]
        self.stat_list = snapshot["stat_list"]
        self.best_valid_epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        print(f"Best validation loss: {self.best_valid_loss}")
        print(f"Best validation epoch: {self.best_valid_epoch}")


    def _run_batch(self, source, targets, files):
        self.optimizer.zero_grad()
        output = self.model(source)
        if self.customized_loss:
            loss, mse = self.criterion(output, targets)
            # print(loss, mse)
        else:
            loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        # try:
        if self.customized_loss:
            return loss.item(), getR2(output.to('cpu'), targets.to('cpu')).item(), torch.sum(targets), torch.sum(output), mse.item()
        else:
            return loss.item(), getR2(output.to('cpu'), targets.to('cpu')).item(), torch.sum(targets), torch.sum(output)
        # except Exception as e:
        #     print(e)
        #     print(files, loss)

    def _train_epoch(self, epoch):
        running_loss = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_R2 = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_foot_sum = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_pred_sum = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_mse_loss = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        
        # start = time.time()
        # print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_DG)}")
        self.train_DG.sampler.set_epoch(epoch)
        self.model.train()
        for data in tqdm(self.train_DG):
            source, targets, _, files = data
            source = source.to(self.local_rank, dtype=torch.float)
            targets = targets.to(self.local_rank, dtype=torch.float)
            if self.customized_loss:
                batch_loss, batch_r2, batch_foot_sum, batch_pred_sum, mse_loss = self._run_batch(source, targets, files)
                running_mse_loss += mse_loss
            else:
                batch_loss, batch_r2, batch_foot_sum, batch_pred_sum = self._run_batch(source, targets, files)
            running_R2 += batch_r2
            running_loss += batch_loss
            running_foot_sum += batch_foot_sum
            running_pred_sum += batch_pred_sum
            
            
        running_loss = running_loss/len(self.train_DG)
        running_R2 = running_R2/len(self.train_DG)
        running_foot_sum = running_foot_sum/len(self.train_DG)
        running_pred_sum = running_pred_sum/len(self.train_DG)
        running_mse_loss = running_mse_loss/len(self.train_DG)
        # print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_DG)} | Training loss: {round(running_loss.item(), 3)} | Training R2: {round(running_R2.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")
        if self.customized_loss:
            return running_loss, running_R2, running_foot_sum, running_pred_sum, running_mse_loss    
        else:
            return running_loss, running_R2, running_foot_sum, running_pred_sum

    def _valid_epoch(self, epoch):
        running_loss = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_custom_loss = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_R2 = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_foot_sums = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        running_foot_pred_sums = torch.tensor(0).to(self.local_rank, dtype=torch.float)
        # print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.valid_DG)}")

        # start = time.time()
        self.valid_DG.sampler.set_epoch(epoch)
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.valid_DG):
                source, targets, _, files = data
                source = source.to(self.local_rank, dtype=torch.float)
                targets = targets.to(self.local_rank, dtype=torch.float)
                output = self.model(source)
                if self.customized_loss:
                    valid_loss_custom, valid_loss = self.criterion(output, targets)
                    running_loss += valid_loss.item() # Can change it to custom loss
                    running_custom_loss += valid_loss_custom.item()
                else:
                    valid_loss = self.criterion(output, targets)
                    running_loss += valid_loss.item()
                running_R2 += getR2(output.to('cpu'), targets.to('cpu')).item()
                running_foot_sums += torch.sum(targets)
                running_foot_pred_sums += torch.sum(output)

        running_loss = running_loss/len(self.valid_DG)
        running_R2 = running_R2/len(self.valid_DG)
        running_foot_sums = running_foot_sums/len(self.valid_DG)
        running_foot_pred_sums = running_foot_pred_sums/len(self.valid_DG)
        running_custom_loss = running_custom_loss/len(self.valid_DG)
        # print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.valid_DG)} | Validation loss: {round(running_loss.item(), 3)} | Validation R2: {round(running_R2.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")
        return running_loss, running_R2, running_foot_sums, running_foot_pred_sums, running_custom_loss

    def train(self, max_epoch:int):
        train_b_sz = len(next(iter(self.train_DG))[0])
        valid_b_sz = len(next(iter(self.valid_DG))[0])
        print("Training begins ....")
        for epoch in range(self.epochs_run, max_epoch):
            start = time.time()
            if self.customized_loss:
                train_loss, train_R2, train_foot_sum, train_pred_sum, train_mse_loss = self._train_epoch(epoch)
            else:
                train_loss, train_R2, train_foot_sum, train_pred_sum = self._train_epoch(epoch)
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            train_loss = train_loss/self.args.world_size

            dist.all_reduce(train_R2, op=dist.ReduceOp.SUM)
            train_R2 = train_R2/self.args.world_size

            dist.all_reduce(train_foot_sum, op=dist.ReduceOp.SUM)
            train_foot_sum = train_foot_sum/self.args.world_size

            dist.all_reduce(train_pred_sum, op=dist.ReduceOp.SUM)
            train_pred_sum = train_pred_sum/self.args.world_size
            
            if self.customized_loss:
                dist.all_reduce(train_mse_loss, op=dist.ReduceOp.SUM)
                train_mse_loss = train_mse_loss/self.args.world_size
                print(f"[HOST {host}] Epoch {epoch} | Batchsize: {train_b_sz} | Steps: {len(self.train_DG)} | Training loss: {round(train_loss.item(), 3)} | Training mse loss: {round(train_mse_loss.item(), 3)}| Training R2: {round(train_R2.item(), 3)} | Foot sum: {round(train_foot_sum.item(), 3)} | Pred sum: {round(train_pred_sum.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")
            else:
                print(f"[HOST {host}] Epoch {epoch} | Batchsize: {train_b_sz} | Steps: {len(self.train_DG)} | Training loss: {round(train_loss.item(), 3)} | Training R2: {round(train_R2.item(), 3)} | Foot sum: {round(train_foot_sum.item(), 3)} | Pred sum: {round(train_pred_sum.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")

            dist.barrier()
            
            if self.valid_DG:
                start = time.time()
                
                # valid_loss, valid_R2 = self._valid_epoch(epoch)
                valid_loss, valid_R2, valid_foot_sum, valid_foot_pred_sum, valid_custom_loss = self._valid_epoch(epoch)
                
                dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
                valid_loss = valid_loss/self.args.world_size

                dist.all_reduce(valid_R2, op=dist.ReduceOp.SUM)
                valid_R2 = valid_R2/self.args.world_size

                dist.all_reduce(valid_foot_sum, op=dist.ReduceOp.SUM)
                valid_foot_sum = valid_foot_sum/self.args.world_size

                dist.all_reduce(valid_foot_pred_sum, op=dist.ReduceOp.SUM)
                valid_foot_pred_sum = valid_foot_pred_sum/self.args.world_size

                dist.all_reduce(valid_custom_loss, op=dist.ReduceOp.SUM)
                valid_custom_loss = valid_custom_loss/self.args.world_size

                if self.customized_loss:
                    print(f"[HOST {host}] Epoch {epoch} | Batchsize: {valid_b_sz} | Steps: {len(self.valid_DG)} | Validation mse loss: {round(valid_loss.item(), 3)} | Validation custom loss: {round(valid_custom_loss.item(), 3)} | Validation R2: {round(valid_R2.item(), 3)} | Foot sum: {round(valid_foot_sum.item(), 3)} | Pred sum: {round(valid_foot_pred_sum.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")
                else:
                    print(f"[HOST {host}] Epoch {epoch} | Batchsize: {valid_b_sz} | Steps: {len(self.valid_DG)} | Validation loss: {round(valid_loss.item(), 3)} | Validation R2: {round(valid_R2.item(), 3)} | Foot sum: {round(valid_foot_sum.item(), 3)} | Pred sum: {round(valid_foot_pred_sum.item(), 3)} | Time taken: {round((time.time()-start)/60, 3)} minutes")
                
                if self.global_rank == 0 and self.location and self.experiment:
                        # self.stat_list.append({"epoch":epoch, "training_loss":round(train_loss.item(), 3), "train_R2":round(train_R2.item(), 3), "validation_loss":round(valid_loss.item(), 3), "validation_R2":round(valid_R2.item(), 3)})
                        self.stat_list.append({"epoch":epoch, "training_loss":round(train_loss.item(), 3), "train_R2":round(train_R2.item(), 3), "validation_loss":round(valid_loss.item(), 3), "validation_R2":round(valid_R2.item(), 3), "target_sum":round(valid_foot_sum.item(), 3), "prediction_sum":round(valid_foot_pred_sum.item(), 3)})
                        pd.DataFrame(self.stat_list).to_csv(f"{self.location}/training_stats_{experiment}.csv", index=False)

                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss.item()
                    self.best_valid_epoch = epoch
                    if self.global_rank == 0 and self.location and self.experiment:
                        # self.stat_list.append({"epoch":epoch, "training_loss":round(train_loss.item(), 3), "train_R2":round(train_R2.item(), 3), "validation_loss":round(valid_loss.item(), 3), "validation_R2":round(valid_R2.item(), 3)})
                        # pd.DataFrame(self.stat_list).to_csv(f"{self.location}/training_stats_{experiment}.csv", index=False)
                        self._save_snapshot(epoch)
                        
                elif epoch - self.best_valid_epoch > self.early_stop_thresh:
                        print(f"Validation loss did not improve in last {self.early_stop_thresh} epochs")
                        print("Early stopped training at epoch %d" % epoch)
                        print(f"Validation loss did not improve in last {self.early_stop_thresh} epochs. Early stopped training at epoch {epoch}")
                        break
                        
                dist.barrier()
                        
            else:
                self.stat_list.append({"epoch":epoch, "training_loss":round(train_loss, 3), "train_R2":round(train_R2, 3)})
                if self.global_rank == 0 and epoch % 10 == 0:
                    self._save_snapshot(epoch)
            
    
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "valid_loss": self.best_valid_loss,
            "stat_list": self.stat_list,
            "args": self.args
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Validation loss: {round(self.best_valid_loss, 3)} | Training snapshot saved at {self.snapshot_path}")


def get_dataloaders():
    transform = args.transform
    batch_size = args.batch_size

    training_file = "training_files/training_CONUS.csv"
    validation_file = "training_files/validation_CONUS.csv"
    test_file = "training_files/test_CONUS.csv"

    train = list(pd.read_csv(training_file)['path'])
    test = list(pd.read_csv(test_file)['path'])
    valid = list(pd.read_csv(validation_file)['path'])
    seed = 42
    random.Random(seed).shuffle(train)
    
    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(valid)
    train = train[:args.train_limit]
    # valid = valid[:100]
    
    print("Training file:", training_file)
    print("Length of training data:", len(train))
    print("Length of validation data:", len(valid))
    print("Length of test data:", len(test))
    
    train_dataset = FootDataset(train, transform=transform, extension='.nc', backhours=args.backhours)
    train_DG = DataLoader(train_dataset,  batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=args.num_workers, sampler=DistributedSampler(train_dataset))

    valid_dataset = FootDataset(valid, transform=transform, extension='.nc', backhours=args.backhours)
    valid_DG = DataLoader(valid_dataset,  batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, num_workers=args.num_workers, sampler=DistributedSampler(valid_dataset))

    test_dataset = FootDataset(test, transform=transform, extension='.nc', backhours=args.backhours)
    test_DG = DataLoader(test_dataset,  batch_size=batch_size,
                                                  shuffle=False, num_workers=8)#, sampler=DistributedSampler(test_dataset))
    return train_DG, valid_DG, test_DG


class FootLoss(nn.Module):
    def __init__(self, alpha=0): #, lamb=100
        super(FootLoss, self).__init__()
        self.alpha = alpha
        print("FootLoss alpha:", self.alpha)
        # self.lamb = lamb
        self.mse = nn.MSELoss()
    def forward(self, output, target):
        mse = self.mse(output, target)
        output_sum = torch.sum(output)
        target_sum = torch.sum(target)
        sum_diff = torch.abs(output_sum - target_sum)
        norm_diff = sum_diff/target_sum
        # combined_loss = (1-self.alpha)*mse + self.alpha*sum_diff
        combined_loss = mse + self.alpha*norm_diff
        return combined_loss, mse
        

def main(args, snapshot_path: str = "snapshot.pt", location="", experiment=""):
    if location and experiment:
        ddp_setup()
        train_DG, valid_DG, test_DG = get_dataloaders()
        # dataset, model, optimizer = load_train_objs()
        # unet = UNet(n_channels=16, n_classes=1)
        if args.backhours == 24:
            unet = NestedUNet(input_channels=49, num_classes=1)
            # unet = UNet(n_channels=24, n_classes=1)
        elif args.backhours == 18:
            unet = NestedUNet(input_channels=20, num_classes=1)
            # unet = UNet(n_channels=20, n_classes=1)
        elif args.backhours == 12:
            unet = NestedUNet(input_channels=16, num_classes=1)
            # unet = UNet(n_channels=16, n_classes=1)
            
        optimizer = optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = optim.NAdam(unet.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr, momentum=0.9)
        if not args.custom_loss:
            criterion = nn.MSELoss()
        else:
            criterion = FootLoss(alpha=args.custom_loss_alpha)
        # print(optimizer)
        # print(unet)
        print(criterion)
        print(args)
        trainer = Trainer(model=unet, optimizer=optimizer, criterion=criterion, snapshot_path=snapshot_path, train_DG=train_DG, valid_DG=valid_DG, location=location, experiment=experiment, early_stop_thresh=args.early_stop_thresh, args=args, customized_loss=args.custom_loss)
        trainer.train(args.total_epochs)
        destroy_process_group()
    else:
        raise Exception (f"location or experiment is not provided")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--experiment', type=str, help='Experiment title for model training')
    parser.add_argument('--transform', default='log-epsilon_xstilt', type=str, help='Transformation for the footprints')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for training')
    parser.add_argument('--total_epochs', default=100, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--early_stop_thresh', default=20, type=int, help='Early stopping criterion (validation loss not improving for given epochs)')
    parser.add_argument('--world_size', type=int, help='World size')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of dataloader workers (default:8)')
    parser.add_argument('--backhours', default=12, type=int, help='Backhours for past meteorology (default: 12h)')
    parser.add_argument('--custom_loss', default='', type=str, help='Custom loss for training (e.g. mass conservation)')
    parser.add_argument('--custom_loss_alpha', default=0, type=float, help='Custom loss weight for training (e.g. mass conservation)')
    parser.add_argument('--train_limit', default=500000, type=int, help='To trim the training data up to a limit')
    args = parser.parse_args()

    # experiment = "experiment3_site_0.5_random_0.5_LongRun_lr1e-4_log_epsilon_e-4_prelu"
    experiment = args.experiment
    location = f"Xmodels/{experiment}/"
    
    if not os.path.exists(location):
        os.makedirs(location)
    snapshot_path = location + f"best_model{experiment}.pth"
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    print("Experiment:", experiment)
    print("Save location:", location)
    print("Transformation (Y):", args.transform)
    print("Batchsize:", args.batch_size)
    print("learning_rate:", args.lr)
    print("Weight decay:", args.weight_decay)
    print("Meteorology backhours:", args.backhours)
    print("Custom loss:", args.custom_loss)
    print("Custom loss alpha:", args.custom_loss_alpha)
    print(f"Snapshot path: {snapshot_path}")
    
    main(args, snapshot_path, location, experiment)
