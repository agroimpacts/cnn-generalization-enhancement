from torch import nn

from ..basicBlocks import basicBlock, bottleNeck


block_num = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3]
}

dilations_by_outStride = {
    16: [1, 1, 1, 2],
    8: [1, 1, 2,4]
}

class Resnet(nn.Module):
    def __init__(self, block, inch, outStride, layers, firstKernel=7, firstStride=2, firstPadding=3):
        super(Resnet, self).__init__()

        dilations = dilations_by_outStride[outStride]

        self.conv1 = nn.Sequential(nn.Conv2d(inch, 64, firstKernel, stride=firstStride, padding=firstPadding), \
                                   nn.BatchNorm2d(64), \
                                   nn.ReLU(True)) # 1/2

        # # 7x7 conv to three 3x3 conv
        # layer0 = []
        # for i in range(ceil(firstKernel/3.0)):
        #     if i == 0:
        #         layer0.append(Conv3x3_bn_relu(inch, 64, padding = 1, stride = firstStride))
        #     else:
        #         layer0.append(Conv3x3_bn_relu(64, 64, padding = 1))
        # self.conv1 = nn.Sequential(*layer0)

        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0) # original
        # self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.stage1 = self.makeStage(block, 64, 1, layers[0], dilation = dilations[0], firstStage=True, ) # 1/4
        self.stage2 = self.makeStage(block, 128, 2, layers[1], dilation = dilations[1])  # 1/8
        self.stage3 = self.makeStage(block, 256, 2, layers[2], dilation = dilations[2])  # 1/16
        self.stage4 = self.makeStage(block, 512, 1, layers[3], dilation = dilations[3]) # 1/16

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        # m.bias.data.fill_(0)

    def makeStage(self, block, transch, firstStride, blocks, dilation, firstStage=False):
        layers = []

        # expansion = 1/4 for basicblock/bottleneck
        outch = int(transch * block.expansion) # 64*4=256

        if firstStage:
            inch = transch  # 64
        else:
            inch = int(transch * block.expansion / 2)

        # dilation setting
        # if dilation is None:
        #     dilations = [1]*blocks
        # else:
        #     if len(dilation) != blocks:
        #         raise ValueError('Expect dilations to have length {}'.format(blocks))

        for i in range(blocks):
            if i == 0:
                conv = block(inch, outch, dilation = dilation, firstStride=firstStride)
            else:
                conv = block(outch, outch, dilation = dilation)
            layers.append(conv)

        return (nn.Sequential(*layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x0 = self.stage1(x) # 128
        x = self.stage2(x0) # 64
        x = self.stage3(x) #32
        x = self.stage4(x)

        return x0, x





class resnet18(nn.Module):
    def __init__(self, inch, outStride):
        super(resnet18, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, outStride, layers=block_num["resnet18"])

    def forward(self, x):
        x = self.resnet(x)
        return x



class resnet34(nn.Module):
    def __init__(self, inch, outStride):
        super(resnet34, self).__init__()
        self.resnet = Resnet(basicBlock, inch, outStride, layers=block_num["resnet34"])

    def forward(self, x):
        x = self.restnet(x)
        return x



class resnet50(nn.Module):
    def __init__(self, inch, outStride):
        super(resnet50, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, outStride, layers=block_num["resnet50"])

    def forward(self, x):
        x = self.resnet(x)
        return x



class resnet101(nn.Module):
    def __init__(self, inch, outStride):
        super(resnet101, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, outStride, layers=block_num["resnet101"])

    def forward(self, x):
        x = self.resnet(x)
        return x



class resnet152(nn.Module):
    def __init__(self, inch, outStride):
        super(resnet152, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, outStride, layers=block_num["resnet152"])

    def forward(self, x):
        x = self.resnet(x)
        return x


