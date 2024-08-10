import math
from torch import nn
import torch.nn.functional as F

from ..basicBlocks import Conv1x1_bn_relu

class GC_bn_relu(nn.Module):
    def __init__(self, inch, outch, kernel, groups = 1):
        super(GC_bn_relu, self).__init__()

        pad = int((kernel - 1) / 2)

        self.right = nn.Sequential(nn.Conv2d(inch, outch, (kernel, 1), padding = (pad, 0), groups=groups),
                                         nn.BatchNorm2d(outch),
                                         nn.ReLU(True),
                                         nn.Conv2d(outch, outch, (1, kernel), padding = (0,pad), groups=groups),
                                         nn.BatchNorm2d(outch),
                                         nn.ReLU(True))
        self.left = nn.Sequential(nn.Conv2d(inch, outch, (1,kernel), padding = (0,pad), groups = groups),
                                        nn.BatchNorm2d(outch),
                                        nn.ReLU(True),
                                        nn.Conv2d(outch, outch, (kernel, 1), padding = (pad, 0), groups=groups),
                                        nn.BatchNorm2d(outch),
                                        nn.ReLU(True))
        self.conv1x1 = Conv1x1_bn_relu(outch, outch, relu=False)
    def forward(self, x):
        out = self.conv1x1(self.right(x) + self.left(x))
        return out


class basicBlock_GC(nn.Module):
    expansion = 1
    def __init__(self, inch, outch, dilation = 1, firstStride = 1):
        super(basicBlock_GC, self).__init__()
        self.firstBlock = (inch != outch)
        transch = outch

        if self.firstBlock:
            self.conv0 = Conv1x1_bn_relu(inch, outch, stride=firstStride, relu=False)

        # 1st 3x3 Conv
        self.conv1 = GC_bn_relu(inch, transch, 3)
        # 2nd 3x3 Conv
        self.conv2 = GC_bn_relu(tranch, outch, 3)


class bottleNeck_GC(nn.Module):
    expansion = 4

    def __init__(self, inch, outch, dialtion=1, firstStride=1, groups = 1, base_width = 64):
        super(bottleNeck_GC, self).__init__()

        self.firstBlock = (inch != outch)
        transch = int(outch / (self.expansion * groups * base_width / 64))

        if self.firstBlock:
            self.conv0 = Conv1x1_bn_relu(inch, outch, stride=firstStride, relu=False)

        # 1x1 conv
        self.conv1 = Conv1x1_bn_relu(inch, transch, stride=firstStride)
        # 3x3 conv
        self.conv2 = GC_bn_relu(transch, transch, 3, groups=groups)
        # 1x1 conv
        self.conv3 = Conv1x1_bn_relu(transch, outch, relu=False)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)

        if self.firstBlock:
            x = self.conv0(x)

        out = self.relu(res + x)

        return out


class BR(nn.Module):
    """
    This is the Boundary Refinement module in Global Convolutional Network
    """
    def __init__(self, inch):
        super(BR, self).__init__()

        self.conv1 = nn.Conv2d(inch, inch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inch, inch, 3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        out = x + res
        return out


class GC(nn.Module):
    """
    This is the Gloable Convolution module in Global Convolutional Network. It converts a large kernel
    convolution into the summation of two pairs of separable convolution.
    """

    def __init__(self, inch, outch, kernel):
        super(GC, self).__init__()

        pad = int((kernel - 1) / 2)

        self.conv_left = nn.Sequential(nn.Conv2d(inch, outch, (kernel, 1), padding=(pad, 0)),
                                       nn.Conv2d(outch, outch, (1, kernel), padding=(0, pad)))
        self.conv_right = nn.Sequential(nn.Conv2d(inch, outch, (1, kernel), padding=(0, pad)),
                                        nn.Conv2d(outch, outch, (kernel, 1), padding=(pad, 0)))

    def forward(self, x):
        xl = self.conv_left(x)
        xr = self.conv_right(x)
        x = xl + xr
        return x



class SEB(nn.Module):
    """
    This is the Semantic Embedding Branch module in EXFuse. It produces a pixel-wise multiplication of all upsampled
    higher level features and the given lower level feature
    """
    def __init__(self, outch, layer = None):
        super(SEB, self).__init__()

        self.outch = outch
        self.Layer = layer

        self.convs = self.makeConvs()


    def makeConvs(self):
        if self.Layer == 0:
            inchs = [64, 256, 512, 1024, 2048]
        elif self.Layer == 1:
            inchs = [256, 512, 1024, 2048]
        else:
            inchs = [self.outch * (2 ** m) for m in range(1, int(math.log2(2048 // self.outch) + 1))]
        # print(inchs)

        convs = []
        for inch in inchs:
            conv = nn.Conv2d(inch, self.outch, 3, padding = 1)
            convs.append(conv)
        return nn.ModuleList(convs)


    def forward(self, x):

        feature = x[-1]

        for i in range(len(self.convs)):
            # print(x[i].size())
            x_high = F.interpolate(self.convs[i](x[i].clone()), size = feature.size()[-2:], mode = "bilinear", align_corners = True)
            feature *= x_high

        return feature


class ECRE(nn.Module):
    """
    This is the Explicit Channel Resolution Embedding module in ExFuse.
    """
    def __init__(self, scale):
        super(ECRE, self).__init__()

        self.ECRE = nn.PixelShuffle(scale)

    def forward(self, x):

        return(self.ECRE(x))


class DAP(nn.Module):
    """
    This is the Densely Adjacent Prediction module in ExFuse. It is used as the last layer of ExFuse, and does the
    prediction using information from adjacent position bounded by specified kernel.
    """
    def __init__(self, kernel):
        super(DAP, self).__init__()

        self.DAP = nn.Sequential(nn.PixelShuffle(kernel),
                                 nn.AvgPool2d(kernel, stride = kernel))

    def forward(self, x):

        return(self.DAP(x))


