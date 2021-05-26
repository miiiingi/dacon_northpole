import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
train = pd.read_csv('data/weekly_train.csv')
test = pd.read_csv('data/public_weekly_test.csv')
train_data_path = 'data/weekly_train/' + train.tail(52*30)['week_file_nm'].values
sample = np.load(train_data_path[-1])
def graph_data(file) : 
    plt.figure(figsize=(15, 5))
    for order in range(file.shape[-1]) : 
        plt.subplot(1, 5, order+1)
        plt.imshow(sample[:, :, order])
    plt.show()
graph_data(sample)