import torch.nn.functional as F

from .resnet_deepLab import *
from ..basicBlocks import Conv1x1_bn_relu
from ..PSP import ASPP


ASPPInchByBackbone = {
    "resnet": 2048
}

class deeplab3(nn.Module):
    def __init__(self, inch, classNum,backbone = resnet50, rates = [1, 6, 12, 18]):
        super(deeplab3, self).__init__()
        # backbone
        self.backbone = backbone(inch)

        # ASPP
        ASPPinch = ASPPInchByBackbone[backbone.__name__.rstrip('0123456789')]
        ASPPoutch = ASPPinch // 8
        self.ASPP = ASPP(ASPPinch, rates = rates, stagech=ASPPoutch)
        self.conv = Conv1x1_bn_relu(ASPPoutch, classNum)


    def forward(self, x):
        x_size = x.size()

        # backbone
        x = self.backbone(x)

        # ASPP
        x = self.ASPP(x)
        x = self.conv(x)

        #upsample
        x = F.interpolate(x, size=x_size[-2:], mode="bilinear", align_corners=True)

        return x