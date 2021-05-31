from numpy.lib.arraypad import pad
from torch import nn as nn
import torch
from part import * 
class Model(nn.Module) :
    def __init__(self, channels_input, channels_output) :
        super(Model, self).__init__()
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downconv_1 = nn.Conv2d(channels_input, channels_input, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downblock_1 = block_residual(channels_input)
        self.downblock_1_act = block_residual(channels_input, activation=True)

        self.downconv_2 = nn.Conv2d(channels_input, channels_input * 2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downblock_2 = block_residual(channels_input * 2)
        self.downblock_2_act = block_residual(channels_input * 2, activation=True)

        self.downconv_3 = nn.Conv2d(channels_input * 2, channels_input * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downblock_3 = block_residual(channels_input * 4)
        self.downblock_3_act = block_residual(channels_input * 4, activation=True)
        
        self.downconv_4 = nn.Conv2d(channels_input * 4, channels_input * 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downblock_4 = block_residual(channels_input* 8)
        self.downblock_4_act = block_residual(channels_input * 8, activation=True)

        self.upconv_3_trans = nn.ConvTranspose2d(channels_input * 8, channels_input * 4, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))   
        self.upconv_3 = nn.Conv2d(channels_input * 8, channels_input * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upblock_3 = block_residual(channels_input * 4)
        self.upblock_3_act = block_residual(channels_input * 4, activation=True)

        self.upconv_2_trans = nn.ConvTranspose2d(channels_input * 4, channels_input * 2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))   
        self.upconv_2 = nn.Conv2d(channels_input * 4, channels_input * 2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upblock_2 = block_residual(channels_input * 2)
        self.upblock_2_act = block_residual(channels_input * 2, activation=True)

        self.upconv_1_trans = nn.ConvTranspose2d(channels_input * 2, channels_input * 1, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))   
        self.upconv_1 = nn.Conv2d(channels_input * 2, channels_input * 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upblock_1 = block_residual(channels_input * 1)
        self.upblock_1_act = block_residual(channels_input * 1, activation=True)

        self.upconv_0_trans = nn.ConvTranspose2d(channels_input * 1, channels_input * 1, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))   
        self.upconv_0 = nn.Conv2d(channels_input * 2, channels_input * 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upblock_0 = block_residual(channels_input * 1)
        self.upblock_0_act = block_residual(channels_input * 1, activation=True)

        self.outputconv = nn.Conv2d(channels_input * 1, channels_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x) : 
        down1 = self.downconv_1(x)
        down1 = self.downblock_1(down1)
        down1 = self.downblock_1_act(down1)
        down1 = self.maxpool(down1)
        down1 = self.dropout(down1)

        down2 = self.downconv_2(down1)
        down2 = self.downblock_2(down2)
        down2 = self.downblock_2_act(down2)
        down2 = self.maxpool(down2)
        down2 = self.dropout(down2)

        down3 = self.downconv_3(down2)
        down3 = self.downblock_3(down3)
        down3 = self.downblock_3_act(down3)
        down3 = self.maxpool(down3)
        down3 = self.dropout(down3)

        down4 = self.downconv_4(down3)
        down4 = self.downblock_4(down4)
        down4 = self.downblock_4_act(down4)
        down4 = self.maxpool(down4)
        down4 = self.dropout(down4)

        up3 = self.upconv_3_trans(down4)
        up3 = torch.cat([down3, up3], dim = 1)
        up3 = self.dropout(up3)
        up3 = self.upconv_3(up3)
        up3 = self.upblock_3(up3)
        up3 = self.upblock_3_act(up3)

        up2 = self.upconv_2_trans(up3)
        up2 = torch.cat([down2, up2], dim = 1)
        up2 = self.dropout(up2)
        up2 = self.upconv_2(up2)
        up2 = self.upblock_2(up2)
        up2 = self.upblock_2_act(up2)

        up1 = self.upconv_1_trans(up2)
        up1 = torch.cat([down1, up1], dim = 1)
        up1 = self.dropout(up1)
        up1 = self.upconv_1(up1)
        up1 = self.upblock_1(up1)
        up1 = self.upblock_1_act(up1)

        up0 = self.upconv_0_trans(up1)
        up0 = torch.cat([x, up0], dim = 1)
        up0 = self.dropout(up0)
        up0 = self.upconv_0(up0)
        up0 = self.upblock_0(up0)
        up0 = self.upblock_0_act(up0)

        return self.outputconv(up0)