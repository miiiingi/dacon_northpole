import pandas as pd
import numpy as np 
from torch.utils.data import Dataset
from torch import nn as nn 
train = pd.read_csv('data/weekly_train.csv')
test = pd.read_csv('data/public_weekly_test.csv')
train_data_path = 'data/weekly_train/' + train.tail(52*30)['week_file_nm'].values
# 테스트 활용 가능 마지막 제공 데이터와 맞춰야 하는 기간 사이에는 2주의 공백이있음.
# 과거 12주의 해빙 변화를 보고 2주 뒤부터 12주 간의 변화를 예측하는 모델을 만들자.
class BatchAct(nn.Module) : 
    def __init__(self, channels_output) -> None:
        super(BatchAct, self).__init__()
        self.batchnorm = nn.BatchNorm2d(channels_output)
        self.relu = nn.ReLU()

    def forward(self, x) : 
        x = self.batchnorm(x)
        return self.relu(x) 

class block_conv(nn.Module) : 
    def __init__(self, channels_input, channels_output, activation = True) : 
        super(block_conv, self).__init__()
        self.conv = nn.Conv2d(channels_input, channels_output, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.activation = activation
        self.channels_output = channels_output
        self.batchnorm = BatchAct(channels_output)
    
    def forward(self, x) : 
        if self.activation : 
            return self.batchnorm(x)
        return x

class block_residual(nn.Module) : 
    def __init__(self, channels, activation = False) : 
        super(block_residual, self).__init__()
        self.activation = activation
        self.batchnorm = BatchAct(channels)
        self.block_conv = block_conv(channels, channels)
        self.block_conv_act = block_conv(channels, channels, activation=False)
    
    def forward(self, x) : 
        x = self.batchnorm(x) 
        out = self.block_conv(x)
        out_ = self.block_conv_act(x)
        out += out_
        if self.activation : 
            return self.batchnorm(out)
        return out 


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

def test_map_func(x_list):
    train_x = []
    for path in x_list:
        train_x.append(np.load(path)[:,:,0:1])
    train_x = np.array(train_x)
    train_x = train_x.astype(np.float32)/250

    return train_x

def make_datasetlist(input_window_size = 12, target_window_size = 12, gap = 2, step = 1) : 
    input_data_list, target_data_list = [], []

    for i in range(0, len(train_data_path)-input_window_size-target_window_size-gap+1, step):
        input_data = train_data_path[i:i+input_window_size]
        target_data = train_data_path[i+input_window_size+gap:i+input_window_size+gap+target_window_size]
        input_data_list.append(input_data)
        target_data_list.append(target_data)

    return input_data_list, target_data_list

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

class dataset_val(Dataset) : 
    def __init__(self) -> None:
        data_input, data_target = make_datasetlist()
        self.list_input = data_input[-52:]
        self.list_target = data_target[-52:]

    def __len__(self) -> int:
        return len(self.list_input)

    def __getitem__(self, index: int):
        data_input, data_target = train_map_func(self.list_input[index], self.list_target[index])
        return data_input, data_target

class dataset_test(Dataset) : 
    def __init__(self) -> None:
        test = pd.read_csv('data/public_weekly_test.csv')
        self.test_path = './data/weekly_train/' + test.tail(12)['week_file_nm']

    def __len__(self) -> int:
        return len(self.test_path)

    def __getitem__(self, index: int):
        data_input = test_map_func(self.test_path[index])
        return data_input

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