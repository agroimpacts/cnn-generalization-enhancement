from ..basicBlocks import Conv3x3_bn_relu
from .separable import *


# dilations for entry, middle, exit flows and last convs
dilations_by_outStride = {
    16: [1, 1, 1, 2],
    8: [1, 2, 2, 4]
}


class Xception(nn.Module):
    def __init__(self, inch, outStride):
        super(Xception, self).__init__()

        dilations = dilations_by_outStride[outStride]
        self.outStride = outStride

        self.conv0 = Conv3x3_bn_relu(inch, 32, padding = 1, stride = 2, relu = True)  # 1/2 256
        self.conv1 = Conv3x3_bn_relu(32, 64, padding = 1, relu = True)

        # Xception flows
        self.entry0 = XceptionBlocks(64, 128, dilation = dilations[0], downsample=True)  # 1/4
        self.entry1 = XceptionBlocks(128, 256, dilation = dilations[0], downsample=(dilations[0] % 2 != 0))  # 1/8
        self.entry2 = XceptionBlocks(256, 728, dilation = dilations[0], downsample=(dilations[1] % 2 != 0))  # 1/16

        self.middle = self.makeMidFlow(728, 728, blocks = 16, dilation = dilations[1])

        self.exit = XceptionBlocks(728, 1024, dilation = dilations[2], downsample = dilations[3] % 2 != 0,
                                   exitflow = True)

        # last 3 convs
        self.conv2 = separable3x3(1024, 1536, dilation = dilations[3], padding = dilations[3])
        self.conv3 = separable3x3(1536, 1536, dilation = dilations[3], padding = dilations[3])
        self.conv4 = separable3x3(1536, 2048, dilation = dilations[3], padding = dilations[3])


    def makeMidFlow(self, inch, outch, blocks, dilation):

        layers = []
        for i in range(blocks):

            block = XceptionBlocks(inch, outch, dilation, downsample = False)
            layers.append(block)

        return(nn.Sequential(*layers))


    def forward(self, x):
        x = self.conv0(x)   # 1/2, 256
        x = self.conv1(x)

        # flows
        x0 = self.entry0(x)  # 1/4, 128

        x = self.entry1(x0)  # 1/8, 64

        x = self.entry2(x)  # 1/16, 32

        x = self.middle(x)

        x = self.exit(x)  # 1/32, 32

        # last 3 convs
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)



        # if self.outStride == 8:
        #     return x0, x
        # elif self.outStride == 16:
        #     return x1, x
        return x0, x


