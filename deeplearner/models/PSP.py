import torch
from torch import nn
from .basicBlocks import Conv3x3_bn_relu, Conv1x1_bn_relu
import torch.nn.functional as F


# PSP structure
# bilinear upsample
class PSP(nn.Module):
    def __init__(self, inch, levels):
        super(PSP, self).__init__()
        # create stages
        self.levels = levels
        self.pspInch = inch

        self.stages = self.makeStages()
        self.conv = Conv3x3_bn_relu(inch * 2, inch, padding = 1)

    def makeStages(self):
        outch = int(self.pspInch/ len(self.levels))
        stages = []
        for level in self.levels:
            if level == 1:
                stage = nn.Sequential(nn.AdaptiveAvgPool2d((level, level)),\
                                      nn.Conv2d(self.pspInch, outch, 1),\
                                      nn.ReLU(True))
            else:
                stage = nn.Sequential(nn.AdaptiveAvgPool2d((level, level)),\
                                      nn.Conv2d(self.pspInch, outch, 1),\
                                      nn.BatchNorm2d(outch),\
                                      nn.ReLU(True))
            stages.append(stage)
        return nn.ModuleList(stages)

    def forward(self, x):
        x_size = x.size()

        x1 = [F.interpolate(stage(x), size = x_size[-2:], mode="bilinear", align_corners=True) \
              for stage in self.stages]
        x1 = torch.cat(x1 + [x], 1)
        x1 = self.conv(x1)

        return x1


# ASPP structure
class ASPP(nn.Module):
    def __init__(self, inch, rates, stagech):
        super(ASPP, self).__init__()
        '''
        This class generates the ASPP module introduced in DeepLabv3: https://arxiv.org/pdf/1706.05587.pdf, which
         concatenates 4 parallel atrous spatial pyramid pooling and the image level features. For more detailed 
         information, please refer to the paper of DeepLabv3
         
         Args:
            inch -- (int) Depth of the input tensor
            rates -- (list) A list of rates of the parallel atrous convolution, including that for the 1x1 convolution
            stagech -- (int) Depth of output tensor for each of the parallel atrous convolution
         
         Returns:
            A tensor after a 1x1 convolution of the concatenated ASPP features
        '''

        # create stages
        self.rates = rates
        self.inch = inch
        self.stagech = stagech

        self.stages = self.makeStages()
        # global feature
        self.globe = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), \
                                   Conv1x1_bn_relu(inch, stagech, relu = False))
        # self.conv1x1 = Conv1x1_bn_relu(inch, stagech, relu = False)
        # self.conv = Conv3x3_bn_relu(inch * 2, inch, padding = 1)
        self.conv = Conv1x1_bn_relu(stagech*(len(rates) + 1), stagech, relu = False)

    def makeStages(self):
        outch = self.stagech
        inch = self.inch
        stages = []
        for rate in self.rates:
            if rate == 1:
                stage = Conv1x1_bn_relu(inch, outch, relu = False)

            else:
                stage = Conv3x3_bn_relu(inch, outch, padding =rate, dilation = rate, relu = False)

            stages.append(stage)
        return nn.ModuleList(stages)

    def forward(self, x):
        x_size = x.size()
        # x1 = [F.interpolate(stage(x), size=x_size[-2:], mode="bilinear", align_corners=True) \
        #       for stage in self.stages]
        x0 = [stage(x) for stage in self.stages]

        # global feature
        x1 = self.globe(x)
        x1 = F.interpolate(x1, size = x_size[-2:], mode = "bilinear", align_corners = True)

        x = torch.cat(x0 + [x1], 1)
        x = self.conv(x)

        return x
