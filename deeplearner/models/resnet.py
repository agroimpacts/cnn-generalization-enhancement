from torch import nn

from .basicBlocks import basicBlock, bottleNeck

block_num = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3]
}


class Resnet(nn.Module):
    def __init__(self, block, inch, layers, firstKernel = 7, firstStride = 2, firstPadding = 3):
        super(Resnet, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv2d(inch, 64, firstKernel, stride=firstStride, padding=firstPadding),\
                                   nn.BatchNorm2d(64),\
                                   nn.ReLU(True))  # 1/2
        self.pool1 = nn.MaxPool2d(3, stride = 2, padding = 0)   # 1/4

        self.stage1 = self.makeStage(block, 64, 1, layers[0], firstStage= True)  # 1/4
        self.stage2 = self.makeStage(block, 128, 2, layers[1])  # 1/8
        self.stage3 = self.makeStage(block, 256, 2, layers[2])
        self.stage4 = self.makeStage(block, 512, 2, layers[3])

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        # m.bias.data.fill_(0)

    def makeStage(self, block, transch, firstStride, blocks, firstStage = False):
        layers = []

        # expansion = 1/4 for basicblock/bottleneck
        outch = int(transch * block.expansion)

        if firstStage:
            inch = transch  # 64
        else:
            inch = int(transch * block.expansion / 2)


        for i in range(blocks):
            if i == 0:
                conv = block(inch, outch, firstStride = firstStride)
            else:
                conv = block(outch, outch)
            layers.append(conv)

        return(nn.Sequential(*layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # x = self.avgpool(x)
        return x


class resnet34(nn.Module):
    def __init__(self, inch):
        super(resnet34, self).__init__()
        self.resnet = Resnet(basicBlock, inch, layers = block_num["resnet34"])
    def forward(self, x):
        x = self.restnet(x)
        return x


class resnet50(nn.Module):
    def __init__(self, inch):
        super(resnet50, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, layers = block_num["resnet50"])
    def forward(self, x):
        x = self.resnet(x)
        return x


class resnet101(nn.Module):
    def __init__(self, inch):
        super(resnet101, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, layers = block_num["resnet101"])
    def forward(self, x):
        x = self.resnet(x)
        return x


class resnet152(nn.Module):
    def __init__(self, inch):
        super(resnet152, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, layers = block_num["resnet152"])
    def forward(self, x):
        x = self.resnet(x)
        return x