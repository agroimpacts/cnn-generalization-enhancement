import torch
from torch import nn

class doublePlus(nn.Module):
    def __init__(self, inch, outch):
        super(doublePlus, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, outch, 3, padding=1), nn.BatchNorm2d(outch))
        self.relu = nn.ReLU(True)
    def forward(self,x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        plus = conv + x
        plus = self.relu(plus)
        return plus


class deepunet(nn.Module):
    def __init__(self, inch, classNum):
        super(deepunet, self).__init__()
        #downsample
        #resnet don;t have bn in first layer of first block, and use 1x1 conv instead

        self.conv0 = nn.Sequential(nn.Conv2d(inch, 32, 3, padding=1), nn.BatchNorm2d(32),nn.ReLU(True)) #640x640
        self.dlyr1 = doublePlus(32,32)
        self.ds1 = nn.MaxPool2d(2, stride=2)  # 320x320
        self.dlyr2 = doublePlus(32,32)
        self.ds2 = nn.MaxPool2d(2, stride=2)   # 160x160
        self.dlyr3 = doublePlus(32, 32)
        self.ds3 = nn.MaxPool2d(2, stride=2)  # 80x80
        self.dlyr4 = doublePlus(32, 32)
        self.ds4 = nn.MaxPool2d(2, stride=2)  # 40x40
        self.dlyr5 = doublePlus(32, 32)
        self.ds5 = nn.MaxPool2d(2, stride=2)  # 20x20
        self.dlyr6 = doublePlus(32, 32)
        self.ds6 = nn.MaxPool2d(2, stride=2)  # 10x10
        self.dlyr7 =doublePlus(32,32)
        #upsample
        self.us7 = nn.ConvTranspose2d(32,32,2,stride =2)  # 20x20
        self.ulyr7 = doublePlus(64,64)
        self.us8 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.ulyr8 = doublePlus(64,64)
        self.us9 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.ulyr9 = doublePlus(64,64)
        self.us10 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.ulyr10 = doublePlus(64,64)
        self.us11 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.ulyr11 = doublePlus(64,64)
        self.us12 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.ulyr12 = doublePlus(64,64)
        #dimension tranformation
        self.dimTrans = nn.Conv2d(64, classNum,1, padding=0)

    def forward(self, x):
        #downsample
        conv0 = self.conv0(x)
        dlyr1 = self.dlyr1(conv0)  # 640x640
        ds1 = self.ds1(dlyr1)
        dlyr2 = self.dlyr2(ds1)  # 320x320
        ds2 = self.ds2(dlyr2)
        dlyr3 = self.dlyr3(ds2)  # 160x160
        ds3 = self.ds3(dlyr3)
        dlyr4 = self.dlyr4(ds3)  # 80x80
        ds4 = self.ds4(dlyr4)
        dlyr5 = self.dlyr5(ds4)  # 40x40
        ds5 =self.ds5(dlyr5)
        dlyr6 = self.dlyr6(ds5)  # 20x20
        ds6 = self.ds6(dlyr6)
        dlyr7 = self.dlyr7(ds6)  # 10x10
        #upsample
        # upIn = torch.cat([dlyr6,downOut],1)  # 10x10, 32-->64
        us7 = self.us7(dlyr7)  # 20x20, 32
        us7 = torch.cat([us7,dlyr6],1)  # 32-->64
        ulyr7 = self.ulyr7(us7)  # 64-->32
        us8 = self.us8(ulyr7)  # 40x40
        us8 = torch.cat([us8,dlyr5],1)  # 32-->64
        ulyr8 = self.ulyr8(us8)   # 64-->32
        us9 = self.us9(ulyr8)  # 80x80
        us9 = torch.cat([us9,dlyr4],1)  # 32-->64
        ulyr9 = self.ulyr9(us9)  # 64-->32
        us10 = self.us10(ulyr9)  # 160x160
        us10 = torch.cat([us10,dlyr3],1)  # 32-->64
        ulyr10 = self.ulyr10(us10)  # 64-->32
        us11 = self.us11(ulyr10)  # 320x320
        us11 = torch.cat([us11,dlyr2],1)  # 32-->64
        ulyr11 = self.ulyr11(us11)  # 64-->32
        us12 = self.us12 (ulyr11)  # 640x640
        us12 = torch.cat([us12, dlyr1],1)  # 32-->64
        ulyr12 = self.ulyr12(us12)

        #dimension transformation and output
        dimTrans = self.dimTrans(ulyr12)
        return dimTrans


class deepunet2(nn.Module):
    def __init__(self, inch, classNum):
        super(deepunet2, self).__init__()
        #downsample
        self.conv0 = nn.Sequential(nn.Conv2d(inch, 32, 3, padding=1), nn.BatchNorm2d(32),nn.ReLU(True)) #640x640
        self.dlyr1 = doublePlus(32,32)
        self.ds1 = nn.MaxPool2d(2, stride=2)  # 320x320
        self.dlyr2 = doublePlus(32,32)
        self.ds2 = nn.MaxPool2d(2, stride=2)   # 160x160
        self.dlyr3 = doublePlus(32, 32)
        self.ds3 = nn.MaxPool2d(2, stride=2)  # 80x80
        self.dlyr4 = doublePlus(32, 32)
        self.ds4 = nn.MaxPool2d(2, stride=2)  # 40x40
        self.dlyr5 = doublePlus(32, 32)
        self.ds5 = nn.MaxPool2d(2, stride=2)  # 20x20
        self.dlyr6 = doublePlus(32, 32)
        self.ds6 = nn.MaxPool2d(2, stride=2)  # 10x10
        self.conv1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.relu = nn.ReLU(True)
        # upsample
        self.us7 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 20x20
        self.ulyr7 = doublePlus(64, 64)
        self.us8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ulyr8 = doublePlus(64, 64)
        self.us9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ulyr9 = doublePlus(64, 64)
        self.us10 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ulyr10 = doublePlus(64, 64)
        self.us11 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ulyr11 = doublePlus(64, 64)
        self.us12 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ulyr12 = doublePlus(64, 64)
        # dimension tranformation
        self.dimTrans = nn.Conv2d(64, classNum, 1, padding=0)

    def forward(self, x):
        # downsample
        conv0 = self.conv0(x)
        dlyr1 = self.dlyr1(conv0)  # 640x640
        ds1 = self.ds1(dlyr1)
        dlyr2 = self.dlyr2(ds1)  # 320x320
        ds2 = self.ds2(dlyr2)
        dlyr3 = self.dlyr3(ds2)  # 160x160
        ds3 = self.ds3(dlyr3)
        dlyr4 = self.dlyr4(ds3)  # 80x80
        ds4 = self.ds4(dlyr4)
        dlyr5 = self.dlyr5(ds4)  # 40x40
        ds5 = self.ds5(dlyr5)
        dlyr6 = self.dlyr6(ds5)  # 20x20
        ds6 = self.ds6(dlyr6)  # 10x10
        dlyr7 = self.conv1(ds6)
        dlyr7 = self.conv2(dlyr7)
        dlyr7 = ds6 + dlyr7 #10x0
        downOut = self.relu(dlyr7)  # 10x10
        # upsample
        upIn = torch.cat([dlyr7,downOut],1)  # 10x10, 32-->64
        us7 = self.us7(upIn)  # 20x20, 32
        us7 = torch.cat([us7, dlyr6], 1)  # 32-->64
        ulyr7 = self.ulyr7(us7)  # 64-->32
        us8 = self.us8(ulyr7)  # 40x40
        us8 = torch.cat([us8, dlyr5], 1)  # 32-->64
        ulyr8 = self.ulyr8(us8)  # 64-->32
        us9 = self.us9(ulyr8)  # 80x80
        us9 = torch.cat([us9, dlyr4], 1)  # 32-->64
        ulyr9 = self.ulyr9(us9)  # 64-->32
        us10 = self.us10(ulyr9)  # 160x160
        us10 = torch.cat([us10, dlyr3], 1)  # 32-->64
        ulyr10 = self.ulyr10(us10)  # 64-->32
        us11 = self.us11(ulyr10)  # 320x320
        us11 = torch.cat([us11, dlyr2], 1)  # 32-->64
        ulyr11 = self.ulyr11(us11)  # 64-->32
        us12 = self.us12(ulyr11)  # 640x640
        us12 = torch.cat([us12, dlyr1], 1)  # 32-->64
        ulyr12 = self.ulyr12(us12)

        # dimension transformation and output
        dimTrans = self.dimTrans(ulyr12)

        return dimTrans
