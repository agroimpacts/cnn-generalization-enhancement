import torch
from torch import nn
from .basicBlocks import doubleConv


# unet construction
class unet(nn.Module):
    def __init__(self, inch, classNum):
        super(unet, self).__init__()
        # downsample
        self.dlyr1 = doubleConv(inch, 64)
        self.ds = nn.MaxPool2d(2, stride=2)
        self.dlyr2 = doubleConv(64, 128)
        self.dlyr3 = doubleConv(128, 256)
        self.dlyr4 = doubleConv(256, 512)
        self.dlyr5 = doubleConv(512, 1024)
        self.dlyr6 = doubleConv(1024, 2048)

        # upsample
        self.us_init = nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1)
        self.ulyr_init = doubleConv(2048, 1024)
        self.us6 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.ulyr6 = doubleConv(1024, 512)  # 512x32x32
        self.us7 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.ulyr7 = doubleConv(512, 256)
        self.us8 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.ulyr8 = doubleConv(256, 128)
        self.us9 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.ulyr9 = doubleConv(128, 64)
        self.dimTrans = nn.Conv2d(64, classNum, 1)

    def forward(self, x):
        # downsample
        dlyr1 = self.dlyr1(x)
        ds1 = self.ds(dlyr1) #

        dlyr2 = self.dlyr2(ds1)
        ds2 = self.ds(dlyr2)
        dlyr3 = self.dlyr3(ds2)
        ds3 = self.ds(dlyr3)
        dlyr4 = self.dlyr4(ds3)
        ds4 = self.ds(dlyr4)
        dlyr5 = self.dlyr5(ds4)
        ds_last = self.ds(dlyr5)
        dlyr_last = self.dlyr6(ds_last)
        # upsample

        us_init = self.us_init(dlyr_last)
        ulyr_init = self.ulyr_init(torch.cat([us_init, dlyr5], 1))
        us6 = self.us6(ulyr_init)
        merge6 = torch.cat([us6, dlyr4], 1)  # channel is the second dimension after batch operation
        ulyr6 = self.ulyr6(merge6)
        us7 = self.us7(ulyr6)
        merge7 = torch.cat([us7, dlyr3], 1)
        ulyr7 = self.ulyr7(merge7)
        us8 = self.us8(ulyr7)
        merge8 = torch.cat([us8, dlyr2], 1)
        ulyr8=self.ulyr8(merge8)
        us9 = self.us9(ulyr8)
        merge9 = torch.cat([us9, dlyr1], 1)
        ulyr9 = self.ulyr9(merge9)
        dimTrans = self.dimTrans(ulyr9)
        
        return dimTrans