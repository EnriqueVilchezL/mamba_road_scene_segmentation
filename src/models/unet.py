import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two unpadded 3x3 convolutions followed by ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Up-conv then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            in_channels, out_channels
        )  # in_channels = skip + upsampled

    def forward(self, x1, x2):
        x1 = self.upconv(x1)

        # Crop x2 to match x1 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = (
            x2[:, :, diffY // 2 : -diffY // 2, diffX // 2 : -diffX // 2]
            if diffX and diffY
            else x2
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024
        x = self.up1(x5, x4)  # 512
        x = self.up2(x, x3)  # 256
        x = self.up3(x, x2)  # 128
        x = self.up4(x, x1)  # 64
        logits = self.outc(x)  # output
        return logits
