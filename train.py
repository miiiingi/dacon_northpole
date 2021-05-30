from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn as nn
from part import * 
from model_unet import Model 
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-paramter
learning_rate = 0.0005
batchsize = 2
num_epochs = 100

traindset = dataset_train()
testset = dataset_test()
trainloader = DataLoader(traindset, batch_size = batchsize)

model = Model(12, 12)
optimizer = Adam(model.parameters(), lr = learning_rate)
criterion = nn.L1Loss()
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
total_step = len(trainloader)
curr_lr = learning_rate 
for image, target in trainloader : 
    images = images.to(device)
    images = images.to(device)