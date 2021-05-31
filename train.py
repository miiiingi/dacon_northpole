from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch import mode, nn as nn
from part import dataset_train, dataset_val 
from model_unet import Model 
import torch
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_epochs", type=int, default=600)
    parser.add_argument("--model", type=str, default='model')
    args = parser.parse_args()
    # Hyper-paramter
    learning_rate = 0.1
    batchsize = 2
    trainset = dataset_train()
    testset = dataset_val()
    trainloader = DataLoader(trainset, batch_size = batchsize)
    testloader = DataLoader(testset, batch_size = batchsize)

    model = Model(12, 12).to(device)
    model.load_state_dict(torch.load('model2.pth'))  # failure
    optimizer = SGD(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
    criterion = nn.L1Loss()
    loss_base = 999

    total_step = len(trainloader)
    for epoch in range(args.num_epochs) : 
        loss_total = 0 
        for iter, (images, target) in enumerate(trainloader) : 
            images = images.to(device)
            target = target.to(device)

            outputs = model(torch.squeeze(images, dim = -1))
            loss = criterion(outputs, torch.squeeze(target))
            optimizer.zero_grad()
            loss.clone().backward()
            loss_total += loss
            optimizer.step()
            scheduler.step()
            if iter % 100 == 0 : 
                print('[{}/{}] [{}/{}] loss : {}'.format(iter, len(trainloader), epoch, args.num_epochs, loss))
        writer.add_scalar('loss/train', loss_total, epoch)
        writer.flush()

        model.eval()
        with torch.no_grad() : 
            loss_comp = 0 
            for iter, (images, target) in enumerate(testloader) : 
                images = images.to(device)
                target = target.to(device)
                
                outputs = model(torch.squeeze(images, dim = -1))
                loss = criterion(outputs, torch.squeeze(target))
                loss_comp += loss

            if loss_comp < loss_base : 
                loss_base = loss_comp
                torch.save(model.state_dict(), '{}.pth'.format(args.model))
        
        writer.add_scalar('loss/validation', loss_comp, epoch)
        writer.flush()
    writer.close()

