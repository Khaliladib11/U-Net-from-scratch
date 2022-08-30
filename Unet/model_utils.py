import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownConv, self).__init__()

        self.double_conv = DoubleConv(input_channels, output_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_cat = self.double_conv(x)
        x = self.pool(x_cat)
        return x, x_cat


class UpConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpConv, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=input_channels,
                                          out_channels=output_channels,
                                          kernel_size=2,
                                          stride=2)
        self.double_conv = DoubleConv(input_channels, output_channels)

    def forward(self, x, x_cat):
        x = self.up_conv(x)
        if x.shape != x_cat.shape:
            x = F.resize(x, size=x_cat.shape[2:])
        x = torch.cat([x, x_cat], dim=1)
        x = self.double_conv(x)
        return x


class OutputConv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
