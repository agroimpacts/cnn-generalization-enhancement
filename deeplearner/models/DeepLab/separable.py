from torch import nn

from ..basicBlocks import Conv1x1_bn_relu


class separable3x3(nn.Module):
    def __init__(self, inch, outch, stride = 1, padding = 1, dilation = 1):
        super(separable3x3, self).__init__()

        # depthwise then pointwise, followed by batch normalization and relu
        self.separable = nn.Sequential(nn.Conv2d(inch, inch, 3, padding = padding, stride = 1, dilation = dilation,
                                                 groups = inch),
                                       nn.BatchNorm2d(inch),
                                       nn.Conv2d(inch, outch, 1, stride = stride),
                                       nn.BatchNorm2d(outch),
                                       nn.ReLU())
    def forward(self, x):
        out = self.separable(x)
        return out


class XceptionBlocks(nn.Module):
    def __init__(self, inch, outch, dilation, downsample = False, exitflow = False):
        super(XceptionBlocks, self).__init__()

        self.skip = downsample or inch != outch

        # conditional values
        transch = inch if exitflow else outch
        lastStride = 2 if downsample else 1

        self.sep0 = separable3x3(inch, transch, padding=dilation, dilation=dilation)
        self.sep1 = separable3x3(transch, outch, padding=dilation, dilation=dilation)
        self.sep2 = separable3x3(outch, outch, padding=dilation, dilation=dilation, stride = lastStride)
        self.relu = nn.ReLU()

        # 1x1 conv if downsample
        self.conv = Conv1x1_bn_relu(inch, outch, stride=lastStride, relu=False)


    def forward(self, x):

        out = self.sep0(x)
        out = self.sep1(out)
        out = self.sep2(out)

        if self.skip:
            x = self.conv(x)

        out += x
        out = self.relu(out)

        return out



