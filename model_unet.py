from numpy.lib.arraypad import pad
from torch import nn as nn
from part import * 
class Model(nn.Module) :
    def __init__(self, channels_input, channels_output) :
        super(Model).__init__()
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downconv_1 = nn.Conv2d(channels_input, channels_input, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downconv_2 = nn.Conv2d(channels_input, channels_input * 2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downconv_3 = nn.Conv2d(channels_input * 2, channels_input * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.downconv_4 = nn.Conv2d(channels_input * 4, channels_input * 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

        self.upsample = nn.Upsample()   
        self.upconv_1 = nn.Conv2d(channels_input * 8, channels_input * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upconv_2 = nn.Conv2d(channels_input * 4, channels_input * 2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.upconv_3 = nn.Conv2d(channels_input * 2, channels_input * 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.outputconv = nn.Conv2d(channels_input * 1, channels_output, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))

    def forward(self, x) : 
        print(x.shape)
        down1 = self.downconv_1(x)
        print(down1.shape)
        down1 = block_residual(down1, self.channels_output)
        down1 = block_residual(down1, self.channels_output, activation=True)
        down1 = self.madown1pool(down1)
        down1 = self.dropout(down1)

        down2 = self.downconv_2(down1)
        print(down2.shape)
        down2 = block_residual(down2, self.channels_output * 2)
        down2 = block_residual(down2, self.channels_output * 2, activation=True)
        down2 = self.madown2pool(down2)
        down2 = self.dropout(down2)

        down3 = self.downconv_3(down2)
        print(down3.shape)
        down3 = block_residual(down3, self.channels_output * 4)
        down3 = block_residual(down3, self.channels_output * 4, activation=True)
        down3 = self.madown3pool(down3)
        down3 = self.dropout(down3)
        
        down4 = self.downconv_4(down3)
        print(down4.shape)
        down4 = block_residual(down4, self.channels_output * 8)
        down4 = block_residual(down4, self.channels_output * 8, activation=True)
        exit()

        up3 = self.upsample(down4)
        up3 = torch.cat()



        return