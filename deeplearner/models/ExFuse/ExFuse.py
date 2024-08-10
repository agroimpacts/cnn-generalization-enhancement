from .resnet_ExFuse import *
from .basicModules import *


class exfuse(nn.Module):
    def __init__(self, inch, classNum, backbone = resnext101, GCkernel = 7, DAPkernel = 3, aux_ECRE = False):
        super(exfuse, self).__init__()

        self.aux_ECRE = aux_ECRE

        # downsample
        self.ds = backbone(inch)
        self.ds1 = nn.Sequential(SEB(256),
                                 GC(256, classNum, GCkernel),
                                 BR(classNum))  #64
        self.ds2 = nn.Sequential(SEB(512),
                                 GC(512, classNum, GCkernel),
                                 BR(classNum))  # 32
        self.ds3 = nn.Sequential(SEB(1024),
                                 GC(1024, classNum, GCkernel),
                                 BR(classNum))  # 16
        self.ds4 = nn.Sequential(GC(2048, classNum, GCkernel),
                                 BR(classNum))  # 8

        # upsample
        if self.aux_ECRE:
            self.us1 = ECRE(2)   # 16
        else:
            self.us1 = nn.ConvTranspose2d(classNum, classNum, 2, stride=2)
        self.us2 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride=2))  # 32
        self.us3 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride=2))  # 64
        self.us4 = nn.Sequential(BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum, 2, stride=2),
                                 BR(classNum),
                                 nn.ConvTranspose2d(classNum, classNum * DAPkernel * DAPkernel, 2, stride=2),
                                 BR(classNum * DAPkernel * DAPkernel),
                                 DAP(DAPkernel))  # 256

    def forward(self, x):
        x1, x2, x3, x4, _, _ = self.ds(x)

        x1 = self.ds1([x2, x3, x4, x1])
        x2 = self.ds2([x3, x4, x2])
        x3 = self.ds3([x4, x3])
        x4 = self.ds4(x4)

        x3_aux = self.us1(x4)
        x3 = x3_aux + x3
        x2 = self.us2(x3) + x2
        x1 = self.us3(x2) + x1
        x1 = self.us4(x1)
        # print(x1.size())

        if self.aux_ECRE:
            # x3_aux -- 1/16 of input size
            return x3_aux,  x1
        else:
            return x1




