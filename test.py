import pandas as pd
from part import dataset_test, dataset_train
from torch.utils.data import DataLoader
from model_unet import Model 
import argparse
import torch
import numpy as np 
# train = pd.read_csv('data/weekly_train.csv')
# test = pd.read_csv('data/public_weekly_test.csv')
# train_data_path = 'data/weekly_train/' + train.tail(52*30)['week_file_nm'].values
# test_data_path = 'data/weekly_train/' + test.tail(12)['week_file_nm'].values

# for i in range(len(np.load(train_data_path[0])[:, :, 0])) :
#     print(max(np.load(train_data_path[0])[:, :, 0][i]))
# print(np.maximum(np.load(train_data_path[0])[:, :, 0]))
# print(trainset)
testset = dataset_test()
testloader = DataLoader(testset, batch_size=12)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_northpole prediction') 
    parser.add_argument("--model", type=str, default='model')
    args = parser.parse_args()
    model = Model(12, 12)
    model.load_state_dict(torch.load('{}.pth'.format(args.model)))  # failure
    for images in testloader : 
        output = model(torch.from_numpy(np.expand_dims(images, axis = 0))).detach().numpy()
    submit = pd.read_csv('data/sample_submission.csv')

    sub_2020 = submit.loc[:11, ['week_start']].copy()
    sub_2021 = submit.loc[12:].copy()
    sub_2020 = pd.concat([sub_2020, pd.DataFrame(output[0].reshape(12, -1) * 250)], axis=1)
    sub_2021.columns = sub_2020.columns
    submission = pd.concat([sub_2020, sub_2021])
    submission.to_csv('sample_submission.csv', index=False, float_format='%.10f')