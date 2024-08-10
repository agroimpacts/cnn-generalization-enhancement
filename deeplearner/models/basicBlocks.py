from torch import nn


class Conv3x3_bn_relu(nn.Module):
    def __init__(self, inch, outch, padding = 0, stride =1, dilation = 1, groups = 1, relu = True):
        super(Conv3x3_bn_relu, self).__init__()
        self.applyRelu = relu

        self.conv = nn.Sequential(nn.Conv2d(inch, outch, 3, padding = padding, stride = stride, dilation = dilation,
                                            groups = groups),
                                  nn.BatchNorm2d(outch))
        if self.applyRelu:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        if self.applyRelu:
            out = self.relu(out)
        return out

class Conv1x1_bn_relu(nn.Module):
    def __init__(self, inch, outch, stride = 1, padding = 0, dilation = 1, groups = 1, relu = True):
        super(Conv1x1_bn_relu, self).__init__()
        self.applyRelu = relu
        self.conv = nn.Sequential(nn.Conv2d(inch, outch, 1, stride = stride, padding = padding, dilation = dilation,
                                            groups = groups),
                                  nn.BatchNorm2d(outch))

        if self.applyRelu:
            self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x.clone())
        if self.applyRelu:
            x = self.relu(x)
        return x


# Consecutive 2 convolution with batch normalization and ReLU activation
class doubleConv(nn.Module):
    def __init__(self, inch, outch):
        super(doubleConv, self).__init__()
        self.conv1 = Conv3x3_bn_relu(inch, outch, padding = 1)
        self.conv2 = Conv3x3_bn_relu(outch, outch, padding = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# skip module
# basic unit of resnet
class basicBlock(nn.Module):
    expansion = 1
    def __init__(self, inch, outch, dilation = 1, firstStride = 1, *kwargs):
        super(basicBlock, self).__init__()
        self.firstBlock = (inch != outch)
        transch = outch

        if self.firstBlock:
            self.conv0 = Conv1x1_bn_relu(inch, outch, stride = firstStride, relu = False)


        # 1st 3x3 Conv
        self.conv1 = Conv3x3_bn_relu(inch, transch, stride =  firstStride, padding = dilation,
                                     dilation = dilation)
        # 2nd 3x3 Conv
        self.conv2 = Conv3x3_bn_relu(transch, outch, padding = dilation, dilation = dilation,
                                        relu = False)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        res = self.conv1(x)
        res = self.conv2(res)
        if self.firstBlock:
            x = self.conv0(x)

        out = self.relu(res + x)
        return out


class bottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inch, outch, dilation = 1, firstStride = 1, groups = 1, base_width = 64):
        super(bottleNeck, self).__init__()

        self.firstBlock = (inch != outch)
        transch = int(outch / (self.expansion * groups * base_width / 64))

        # downsample in first 1x1 convolution
        if self.firstBlock:
            self.conv0 = Conv1x1_bn_relu(inch, outch, stride=firstStride, relu=False)

        # 1x1 conv
        self.conv1 = Conv1x1_bn_relu(inch, transch, stride=firstStride)
        # 3x3 conv
        self.conv2 = Conv3x3_bn_relu(transch, transch, padding = dilation, dilation = dilation, groups = groups)
        # 1x1 conv
        self.conv3 = Conv1x1_bn_relu(transch, outch, relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)

        if self.firstBlock:
            x = self.conv0(x)

        out = self.relu(res + x)
        return out


class ConvBlock(nn.Module):
    r"""This module creates a user-defined number of conv+BN+ReLU layers.
    Args:
        in_channels (int): number of input features.
        out_channels (int): number of output features.
        num_conv_layers (int): Number of conv+BN+ReLU layers in the block.
        drop_rate (float): dropout rate at the end of the block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, num_conv_layers=2, drop_rate=0):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True), ]

        # This part has a dynamic size regarding the number of conv layers in the block.
        layers += [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, dilation=dilation, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ] * (num_conv_layers - 1)

        if drop_rate > 0 and num_conv_layers > 1:
            layers += [nn.Dropout(drop_rate)]

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class SeLayer(nn.Module):

    def __init__(self, in_channels, reduction):
        super(SeLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ErrCorrBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ErrCorrBlock, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(out_channels))

        middle_ch = in_channels // reduction

        self.triple_conv = nn.Sequential(

            nn.Conv2d(in_channels, middle_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_ch, middle_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_ch, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()
        self.se = SeLayer(out_channels, reduction)

    def forward(self, x):
        residual = self.conv0(x)

        out = self.triple_conv(x)
        #out = self.se(out)

        out = self.relu(out + residual)

        return out
