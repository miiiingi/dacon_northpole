import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.nn import BatchNorm2d
from torch import nn as nn 
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
train = pd.read_csv('data/weekly_train.csv')
test = pd.read_csv('data/public_weekly_test.csv')
train_data_path = 'data/weekly_train/' + train.tail(52*30)['week_file_nm'].values
# 테스트 활용 가능 마지막 제공 데이터와 맞춰야 하는 기간 사이에는 2주의 공백이있음.
# 과거 12주의 해빙 변화를 보고 2주 뒤부터 12주 간의 변화를 예측하는 모델을 만들자.
def train_map_func(x_list, y_list):
    train_x, train_y = [], []
    for path in x_list:
        train_x.append(np.load(path)[:,:,0:1])
    for path in y_list:
        train_y.append(np.load(path)[:,:,0:1])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    train_x = train_x.astype(np.float32)/250
    train_y = train_y.astype(np.float32)/250
    return train_x, train_y

def make_datasetlist(input_window_size = 12, target_window_size = 12, gap = 2, step = 1) : 
    input_data_list, target_data_list = [], []

    for i in range(0, len(train_data_path)-input_window_size-target_window_size-gap+1, step):
        input_data = train_data_path[i:i+input_window_size]
        target_data = train_data_path[i+input_window_size+gap:i+input_window_size+gap+target_window_size]
        input_data_list.append(input_data)
        target_data_list.append(target_data)

    return input_data_list, target_data_list

def BatchAct(self, x) : 
    x = BatchNorm2d(x)
    return nn.ReLU(x)

def block_conv(self, channels_input, channels_output, activation = True) : 
    x = nn.Conv2d(channels_input, channels_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
    if activation :
        x = BatchAct(x)
    return x

def block_residual(self, x, channels, activation = False) : 
    x = BatchAct(x) 
    x = block_conv(x, channels, channels)
    x_ = block_conv(x, channels, channels, activation = False)
    x = torch.cat([x, x_], dim = 1)
    if activation : 
        return BatchAct(x)
    return x

class dataset_train(Dataset) : 
    def __init__(self) -> None:
        data_input, data_target = make_datasetlist()
        self.list_input = data_input[:-52]
        self.list_target = data_target[:-52]

    def __len__(self) -> int:
        return len(self.list_input)

    def __getitem__(self, index: int):
        data_input, data_target = train_map_func(self.list_input[index], self.list_target[index])
        return data_input, data_target

class dataset_test(Dataset) : 
    def __init__(self) -> None:
        data_input, data_target = make_datasetlist()
        self.list_input = data_input[-52:]
        self.list_target = data_target[-52:]

    def __len__(self) -> int:
        return len(self.list_input)

    def __getitem__(self, index: int):
        data_input, data_target = train_map_func(self.list_input[index], self.list_target[index])
        return data_input, data_target

def mae_score(true, pred):
    score = np.mean(np.abs(true-pred))
    return score

def f1_score(true, pred):
    target = np.where((true > 0.05) & (true < 0.5))
    
    true = true[target]
    pred = pred[target]
    
    true = np.where(true < 0.15, 0, 1)
    pred = np.where(pred < 0.15, 0, 1)
    
    right = np.sum(true * pred == 1)
    precision = right / np.sum(true + 1e-8)
    recall = right / np.sum(pred + 1e-8)

    score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return score

def mae_over_f1(true, pred):
    mae = mae_score(true, pred)
    f1 = f1_score(true, pred)
    score = mae/(f1 + 1e-8)
    
    return score

# def val_score(inp, targ):
#     output = model(inp)
#     score = mae_over_f1(targ.numpy(), output.numpy())
#     return score

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = last_epoch
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr