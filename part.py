import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import xarray 
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
