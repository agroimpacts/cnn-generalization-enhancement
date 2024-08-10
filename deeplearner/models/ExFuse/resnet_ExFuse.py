from torch import nn

from ..basicBlocks import basicBlock, bottleNeck
from .basicModules import bottleNeck_GC

block_num = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
    "exfuse101": [8, 8, 9, 8]
}


class Resnet(nn.Module):
    '''This is the Resnet proposed by Kaiming He but modified to fit the GCN structure, in which max pooling kernel
    is 2 and the final average pooling is removed'''
    def __init__(self, block, inch, layers, groups=1, width=64, firstKernel = 7, firstStride = 2, firstPadding = 3):
        super(Resnet, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv2d(inch, 64, firstKernel, stride=firstStride, padding=firstPadding),\
                                   nn.BatchNorm2d(64),\
                                   nn.ReLU(inplace = False))

        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)  # 1/4

        self.stage1 = self.makeStage(block, 64, 1, layers[0], groups, width, firstStage=True)  # 1/4
        self.stage2 = self.makeStage(block, 128, 2, layers[1], groups, width)  # 1/8
        self.stage3 = self.makeStage(block, 256, 2, layers[2], groups, width)
        self.stage4 = self.makeStage(block, 512, 2, layers[3], groups, width)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        # m.bias.data.fill_(0)

    def makeStage(self, block, transch, firstStride, blocks, groups, base_width, firstStage = False):
        layers = []

        # expansion = 1/4 for basicblock/bottleneck
        outch = int(transch * block.expansion)

        if firstStage:
            inch = transch  # 64
        else:
            inch = int(transch * block.expansion / 2)

        for i in range(blocks):
            if i == 0:
                conv = block(inch, outch, firstStride = firstStride, groups = groups, base_width = base_width)
            else:
                conv = block(outch, outch, groups = groups, base_width = base_width)
            layers.append(conv)

        return(nn.Sequential(*layers))


    def forward(self, x):

        x1 = self.conv1(x.clone())
        x2 = self.pool1(x1.clone())
        stage1 = self.stage1(x2)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        return stage1, stage2, stage3, stage4, x1, x


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

class resnext50(nn.Module):
    def __init__(self, inch):
        super(resnext50, self).__init__()
        width = 4
        groups = 32

        self.resnext = Resnet(bottleNeck, inch, layers=block_num["resnet50"], groups=groups, width=width)
    def forward(self, x):
        x = self.resnext(x)
        return x

class resnext101(nn.Module):
    def __init__(self, inch):
        super(resnext101, self).__init__()
        width = 4
        groups = 32

        self.resnext = Resnet(bottleNeck, inch, layers=block_num["resnet101"], groups=groups, width=width)

    def forward(self, x):
        x = self.resnext(x)
        return (x)


class resnext152(nn.Module):
    def __init__(self, inch):
        super(resnext152,self).__init__()
        width = 4
        groups = 32

        self.resnext = Resnet(bottleNect, inch, layers=block_num["resnet152"], groups=groups, width=width)

    def forward(self, x):
        x = self.resnext(x)
        return x

class exfuse_resnet101(nn.Module):
    def __init__(self, inch):
        super(exfuse_resnet101, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, layers=block_num["exfuse101"])

    def forward(self, x):
        x = self.resnet(x)
        return x


class exfuse_resnext101(nn.Module):
    def __init__(self, inch):
        super(exfuse_resnext101, self).__init__()
        self.resnet = Resnet(bottleNeck, inch, layers=block_num["exfuse101"])

    def forward(self, x):
        x = self.resnet(x)
        return x

class resnet101_GC(nn.Module):
    def __init__(self, inch):
        super(resnet101_GC, self).__init__()
        self.resnet = Resnet(bottleNeck_GC, inch, layers=block_num["resnet101"])
    def forward(self, x):
        x = self.resnet(x)
        return x

class resnext101_GC(nn.Module):
    def __init__(self, inch):
        super(resnext101_GC, self).__init__()
        self.resnext = Resnet(bottleNeck_GC, inch, layers=block_num["resnet101"])
    def forward(self, x):
        x = self.resnext(x)
        return x
