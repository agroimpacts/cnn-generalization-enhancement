import torch
from torch import nn
from torch.nn import functional as F

from .basicBlocks import Conv3x3_bn_relu
from .resnet import resnet50

pspInchByBackbone = {
    "resnet": 2048
}



class pspnet (nn.Module):
    def __init__(self, inch, classNum, pspInch = pspInchByBackbone["resnet"], backbone = resnet50, levels = [1, 2, 3, 6]):
        super(pspnet, self).__init__()
        self.pspInch = pspInch
        self.levels = levels

        self.backbone = backbone(inch)

        self.stages = self.makeStages()
        self.conv1 = Conv3x3_bn_relu(pspInch*2, 512, padding = 1)
        self.conv2 = nn.Conv2d(512, classNum, 1)


    def makeStages(self):
        outch = int(self.pspInch / len(self.levels))
        stages = []
        for level in self.levels:
            if level != 1:
                stage = nn.Sequential(nn.AdaptiveAvgPool2d((level, level)),\
                                  nn.Conv2d(self. pspInch, outch, 1),\
                                  nn.BatchNorm2d(outch),\
                                  nn.ReLU(True))
            else:
                stage = nn.Sequential(nn.AdaptiveAvgPool2d((level, level)),\
                                      nn.Conv2d(self.pspInch, outch, 1),\
                                      nn.ReLU(True))
            stages.append(stage)
        return nn.ModuleList(stages)


    def forward(self, x):
        x_size = x.size()

        x = self.backbone(x)
        x1 = [F.interpolate(stage(x), size=(x.size()[-2:]), mode = "bilinear", align_corners = True)
              for stage in self.stages]
        x1 = torch.cat(x1 + [x], 1)

        # Final upsample
        x1 = F.interpolate(x1, size = x_size[-2: ], mode = "bilinear", align_corners = True)

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        return x1