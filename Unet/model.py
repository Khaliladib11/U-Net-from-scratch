import torch
import torch.nn as nn
from model_utils import DoubleConv, DownConv, UpConv, OutputConv


class UNET(nn.Module):

    def __init__(self, input_channels=3, output_channels=1):
        super(UNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder part
        self.block1 = DownConv(input_channels, 64)
        self.block2 = DownConv(64, 128)
        self.block3 = DownConv(128, 256)
        self.block4 = DownConv(256, 512)

        self.conv_bottleneck = DoubleConv(512, 1024)

        # decoder part
        self.up_1 = UpConv(1024, 512)
        self.up_2 = UpConv(512, 256)
        self.up_3 = UpConv(256, 128)
        self.up_4 = UpConv(128, 64)

        self.final_conv = OutputConv(64, output_channels)

    def forward(self, x):
        x1, x1_cat = self.block1(x)  # skip connection number 1
        x2, x2_cat = self.block2(x1)  # skip connection number 2
        x3, x3_cat = self.block3(x2)  # skip connection number 3
        x4, x4_cat = self.block4(x3)  # skip connection number 4

        bottle_neck = self.conv_bottleneck(x4)

        x5 = self.up_1(bottle_neck, x4_cat)
        x6 = self.up_2(x5, x3_cat)
        x7 = self.up_3(x6, x2_cat)
        x8 = self.up_4(x7, x1_cat)

        output = self.final_conv(x8)

        return output
