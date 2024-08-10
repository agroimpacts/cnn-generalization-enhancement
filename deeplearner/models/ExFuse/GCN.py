from .resnet_ExFuse import *
from .basicModules import *


class gcn(nn.Module):
    def __init__(self, inch, classNum, backbone = resnet101, kernel=7):
        super(gcn, self).__init__()

        # downsample
        self.ds = backbone(inch)
        self.ds1 = nn.Sequential(GC(256, classNum, kernel),
                                 BR(classNum))  # 64
        self.ds2 = nn.Sequential(GC(512, classNum, kernel),
                                 BR(classNum))  # 32
        self.ds3 = nn.Sequential(GC(1024, classNum, kernel),
                                 BR(classNum))  # 16
        self.ds4 = nn.Sequential(GC(2048, classNum, kernel),
                                 BR(classNum))  # 8

        # upsample
        self.us1 = nn.ConvTranspose2d(classNum, classNum, 2, stride=2)  # 16
        self.us2 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride=2))  # 32
        self.us3 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride = 2))  # 64
        self.us4 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride = 2),
                                 BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride = 2),
                                 BR(classNum))  # 256

        # # Kaiming Initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        #         # nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        #         m.bias.data.fill_(1e-9)


    def forward(self, x):

        x1, x2, x3, x4, _, _= self.ds(x)
        x1 = self.ds1(x1)
        x2 = self.ds2(x2)
        x3 = self.ds3(x3)
        x4 = self.ds4(x4)

        x3 = self.us1(x4) + x3  # 1/16
        x2 = self.us2(x3) + x2  # 1/8
        x1 = self.us3(x2) + x1  # 1/4
        x1 = self.us4(x1)
        return x1






