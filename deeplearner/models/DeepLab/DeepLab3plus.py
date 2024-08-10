import torch.nn.functional as F
import torch

from ..PSP import ASPP
from .resnet_deepLab import *
from .xception import *


ASPPInchByBackbone = {
    "resnet": 2048,
    "Xception": 2048
}

quaterOutchByBackbone = {
    "resnet": 256,
    "Xception": 128
}

class deeplab3plus(nn.Module):
    def __init__(self, inch, classNum, backbone = Xception, outStride = 16, rates = [1, 6, 12, 18]):
        super(deeplab3plus, self).__init__()

        # backbone
        self.backbone = backbone(inch, outStride = outStride)

        # ASPP
        ASPPinch = ASPPInchByBackbone[backbone.__name__.rstrip('0123456789')]
        ASPPoutch = ASPPinch // 8
        self.ASPP = ASPP(ASPPinch, rates=rates, stagech=ASPPoutch)

        # decoder
        quaterOutch = quaterOutchByBackbone[backbone.__name__.rstrip('0123456789')]
        self.conv0 = Conv1x1_bn_relu(quaterOutch, ASPPoutch) # 1/4 of origin
        self.last_conv = nn.Sequential(Conv3x3_bn_relu(ASPPoutch*2, 256, 1),
                                       Conv3x3_bn_relu(256, 256, 1),
                                       nn.Conv2d(256, classNum, 1))

    def forward(self, x):

        x_size = x.size()
        x0, x = self.backbone(x)
        inter_size = x0.size()
        x = self.ASPP(x)

        # decoder
        x0 = self.conv0(x0)
        x = F.interpolate(x, size = inter_size[-2:], mode = "bilinear", align_corners=True)
        x = torch.cat([x, x0], 1)
        x = self.last_conv(x)
        x = F.interpolate(x, size = x_size[-2:], mode = "bilinear", align_corners = True)

        return x


class deeplab3plus2(nn.Module):
    def __init__(self, inch, classNum, backbone = resnet101, outStride = 16, rates = [1, 6, 12, 18]):
        super(deeplab3plus2, self).__init__()

        # backbone
        self.backbone = backbone(inch, outStride = outStride)

        # ASPP
        ASPPinch = ASPPInchByBackbone[backbone.__name__.rstrip('0123456789')]
        ASPPoutch = ASPPinch // 8
        self.ASPP = ASPP(ASPPinch, rates=rates, stagech=ASPPoutch)

        # decoder
        quaterOutch = quaterOutchByBackbone[backbone.__name__.rstrip('0123456789')]
        self.conv0 = Conv1x1_bn_relu(quaterOutch, ASPPoutch) # 1/4 of origin
        self.up1 = nn.ConvTranspose2d(256, 256, 6, stride=4, padding=1)
        self.last_conv = nn.Sequential(Conv3x3_bn_relu(ASPPoutch*2, 256, 1),
                                       Conv3x3_bn_relu(256, 256, 1),
                                       nn.Conv2d(256, classNum, 1))
        self.up2 = nn.ConvTranspose2d(classNum, classNum, 6, stride=4, padding=1)


    def forward(self, x):

        x0, x = self.backbone(x)
        x = self.ASPP(x)

        # decoder
        x0 = self.conv0(x0)
        x = self.up1(x)
        x = torch.cat([x, x0], 1)
        x = self.last_conv(x)
        x = self.up2(x)

        return x


