import torch
from torch import nn
from .basicBlocks import ConvBlock, ErrCorrBlock

class DenselyFusedUNet(nn.Module):

    def __init__(self, n_classes, in_channels):
        super(DenselyFusedUNet, self).__init__()

        self.n_classes = n_classes
        self.in_channels = in_channels

        filters = (64, 128, 256, 512, 1024)

        ## -------------Encoder--------------
        self.block1 = ConvBlock(self.in_channels, filters[0], num_conv_layers=3)  # [B, 64, H, W]
        self.block2 = ConvBlock(filters[0], filters[1], num_conv_layers=2)  # [B, 128, H/2, W/2]
        self.block3 = ConvBlock(filters[1], filters[2], num_conv_layers=2)  # [B, 256, H/4, W/4]
        self.block4 = ConvBlock(filters[2], filters[3], num_conv_layers=2)  # [B, 512, H/8, W/8]
        self.block5 = ConvBlock(filters[3], filters[4], num_conv_layers=2)  # [B, 1024, H/16, W/16]
        self.block6 = ConvBlock(filters[4], filters[4], num_conv_layers=1)  # [B, 1024, H/32, W/32]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## -------------Decoder--------------
        '''stage 5d'''
        # h1->256*256, hd5->16*16, Pooling 16 times
        self.h1_PT_hd5 = nn.MaxPool2d(16, 16, ceil_mode=True)
        self.h1_PT_hd5_conv = ConvBlock(filters[0], filters[1], num_conv_layers=1)

        # h2->128*128, hd5->16*16, Pooling 8 times
        self.h2_PT_hd5 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h2_PT_hd5_conv = ConvBlock(filters[1], filters[1], num_conv_layers=1)

        # h3->64*64, hd5->16*16, Pooling 4 times
        self.h3_PT_hd5 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h3_PT_hd5_conv = ConvBlock(filters[2], filters[1], num_conv_layers=1)

        # h4->32*32, hd5->16*16, Pooling 2 times
        self.h4_PT_hd5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h4_PT_hd5_conv = ConvBlock(filters[3], filters[1], num_conv_layers=1)

        # h5->16*16, hd5->16*16, Concatenation
        self.h5_Cat_hd5_conv = ConvBlock(filters[4], filters[3], num_conv_layers=1)

        # hd6->8*8, hd5->16*16, Upsample 2 times
        self.hd6_UT_hd5 = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=4, stride=2, padding=1, dilation=1)

        # fusion(h1_PT_hd5, h2_PT_hd5, h3_PT_hd5, h4_PT_hd5, h5_Cat_hd5, hd6_UT_hd4)
        self.conv5d_1 = ErrCorrBlock(filters[4] * 2, filters[4], reduction=64)
        self.conv5d_2 = ConvBlock(filters[4], filters[4], kernel_size=3, num_conv_layers=1)

        '''stage 4d'''
        # h1->256*256, hd4->32*32, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = ConvBlock(filters[0], filters[0], num_conv_layers=1)

        # h2->128*128, hd4->32*32, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = ConvBlock(filters[1], filters[0], num_conv_layers=1)

        # h3->64*64, hd4->32*32, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = ConvBlock(filters[2], filters[0], num_conv_layers=1)

        # h4->32*32, hd4->32*32, Concatenation
        self.h4_Cat_hd4_conv = ConvBlock(filters[3], filters[2], num_conv_layers=1)

        # hd5->16*16, hd4->32*32, Upsample 2 times
        self.hd5_UT_hd4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=4, stride=2, padding=1, dilation=1)

        # hd6->8*8, hd4->32*32, Upsample 4 times
        self.hd6DimreductSt4 = nn.Conv2d(filters[4], filters[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.hd6_UT_hd4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hd6_UT_hd4_conv = ConvBlock(filters[0], filters[0], num_conv_layers=1)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4, hd6_UT_hd5)
        self.conv4d_1 = ErrCorrBlock(filters[4], filters[3], reduction=32)
        self.conv4d_2 = ConvBlock(filters[3], filters[3], kernel_size=3, num_conv_layers=1)

        '''stage 3d'''
        # h1->256*256, hd3->64*64, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = ConvBlock(filters[0], filters[0] // 2, num_conv_layers=1)

        # h2->128*128, hd3->64*64, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = ConvBlock(filters[1], filters[0] // 2, num_conv_layers=1)

        # h3->64*64, hd3->64*64, Concatenation
        self.h3_Cat_hd3_conv = ConvBlock(filters[2], filters[1], num_conv_layers=1)

        # hd4->32*32, hd3->64*64, Upsample 2 times
        self.hd4_UT_hd3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=4, stride=2, padding=1, dilation=1)

        # hd5->16*16, hd3->64*64, Upsample 4 times
        self.hd5DimreductSt3 = nn.Conv2d(filters[4], filters[0] // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hd5_UT_hd3_conv = ConvBlock(filters[0] // 2, filters[0] // 2, num_conv_layers=1)

        # hd6->8*8, hd3->64*64, Upsample 8 times
        self.hd6DimreductSt3 = nn.Conv2d(filters[4], filters[0] // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd6_UT_hd3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.hd6_UT_hd3_conv = ConvBlock(filters[0] // 2, filters[0] // 2, num_conv_layers=1)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3, hd6_UT_hd3)
        self.conv3d_1 = ErrCorrBlock(filters[3], filters[2], reduction=16)
        self.conv3d_2 = ConvBlock(filters[2], filters[2], kernel_size=3, num_conv_layers=1)

        '''stage 2d '''
        # h1->256*256, hd2->128*128, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = ConvBlock(filters[0], filters[0] // 4, num_conv_layers=1)

        # h2->128*128, hd2->128*128, Concatenation
        self.h2_Cat_hd2_conv = ConvBlock(filters[1], filters[0], num_conv_layers=1)

        # hd3->64*64, hd2->128*128, Upsample 2 times
        self.hd3_UT_hd2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=4, stride=2, padding=1, dilation=1)

        # hd4->32*32, hd2->128*128, Upsample 4 times
        self.hd4DimreductSt2 = nn.Conv2d(filters[3], filters[0] // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hd4_UT_hd2_conv = ConvBlock(filters[0] // 4, filters[0] // 4, num_conv_layers=1)

        # hd5->16*16, hd2->128*128, Upsample 8 times
        self.hd5DimreductSt2 = nn.Conv2d(filters[4], filters[0] // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.hd5_UT_hd2_conv = ConvBlock(filters[0] // 4, filters[0] // 4, num_conv_layers=1)

        # hd6->8*8, hd2->128*128, Upsample 16 times
        self.hd6DimreductSt2 = nn.Conv2d(filters[4], filters[0] // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd6_UT_hd2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.hd6_UT_hd2_conv = ConvBlock(filters[0] // 4, filters[0] // 4, num_conv_layers=1)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2, hd6_UT_hd2)
        self.conv2d_1 = ErrCorrBlock(filters[2], filters[1], reduction=8)
        self.conv2d_2 = ConvBlock(filters[1], filters[1], kernel_size=3, num_conv_layers=1)

        '''stage 1d'''
        # h1->256*256, hd1->256*256, Concatenation
        self.h1_Cat_hd1_conv = ConvBlock(filters[0], filters[0] // 2, num_conv_layers=1)

        # hd2->128*128, hd1->256*256, Upsample 2 times
        self.hd2_UT_hd1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=4, stride=2, padding=1, dilation=1)

        # hd3->64*64, hd1->256*256, Upsample 4 times
        self.hd3DimreductSt1 = nn.Conv2d(filters[2], filters[0] // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hd3_UT_hd1_conv = ConvBlock(filters[0] // 8, filters[0] // 8, num_conv_layers=1)

        # hd4->32*32, hd1->256*256, Upsample 8 times
        self.hd4DimreductSt1 = nn.Conv2d(filters[3], filters[0] // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.hd4_UT_hd1_conv = ConvBlock(filters[0] // 8, filters[0] // 8, num_conv_layers=1)

        # hd5->16*16, hd1->256*256, Upsample 16 times
        self.hd5DimreductSt1 = nn.Conv2d(filters[4], filters[0] // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.hd5_UT_hd1_conv = ConvBlock(filters[0] // 8, filters[0] // 8, num_conv_layers=1)

        # hd6->8*8, hd1->256*256, Upsample 32 times
        self.hd6DimreductSt1 = nn.Conv2d(filters[4], filters[0] // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.hd6_UT_hd1 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.hd6_UT_hd1_conv = ConvBlock(filters[0] // 8, filters[0] // 8, num_conv_layers=1)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1, hd6_UT_hd1)
        self.conv1d_1 = ErrCorrBlock(filters[1], filters[0], reduction=4)
        self.conv1d_2 = ConvBlock(filters[0], filters[0], kernel_size=3, num_conv_layers=1)

        # output
        self.outconv1 = nn.Conv2d(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        ## -------------Encoder-------------
        # set_trace()
        h1 = self.block1(inputs)  # [B, 64, 256, 256]

        h2 = self.maxpool(h1)  # [B, 64, 128, 128]
        h2 = self.block2(h2)  # [B, 128, 128, 128]

        h3 = self.maxpool(h2)  # [B, 128, 64, 64]
        h3 = self.block3(h3)  # [B, 256, 64, 64]

        h4 = self.maxpool(h3)  # [B, 256, 32, 32]
        h4 = self.block4(h4)  # [B, 512, 32, 32]

        h5 = self.maxpool(h4)  # [B, 512, 16, 16]
        h5 = self.block5(h5)  # [B, 1024, 16, 16]

        h6 = self.maxpool(h5)  # [B, 1024, 8, 8]
        hd6 = self.block6(h6)  # [B, 1536, 8, 8]

        ## -------------Decoder-------------
        # stage 5d
        h1_PT_hd5 = self.h1_PT_hd5_conv(self.h1_PT_hd5(h1))  # [5, 128, 16, 16]
        h2_PT_hd5 = self.h2_PT_hd5_conv(self.h2_PT_hd5(h2))  # [5, 128, 16, 16]
        h3_PT_hd5 = self.h3_PT_hd5_conv(self.h3_PT_hd5(h3))  # [5, 128, 16, 16]
        h4_PT_hd5 = self.h4_PT_hd5_conv(self.h4_PT_hd5(h4))  # [5, 128, 16, 16]
        h5_Cat_hd5 = self.h5_Cat_hd5_conv(h5)  # [5, 512, 16, 16]
        hd6_UT_hd5 = self.hd6_UT_hd5(hd6)  # [5, 1024, 16, 16]
        # Conv layer:[B, 2048, 16, 16]
        hd5 = self.conv5d_2(self.conv5d_1(
            torch.cat((h1_PT_hd5, h2_PT_hd5, h3_PT_hd5, h4_PT_hd5, h5_Cat_hd5, hd6_UT_hd5), 1)))  # [B, 1024, 16, 16]

        # stage 4d
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))  # [B, 64, 32, 32]
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))  # [B, 64, 32, 32]
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))  # [B, 64, 32, 32]
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)  # [B, 256, 32, 32]
        hd5_UT_hd4 = self.hd5_UT_hd4(hd5)  # [B, 512, 32, 32]
        hd6_UT_hd4 = self.hd6_UT_hd4_conv(self.hd6_UT_hd4(self.hd6DimreductSt4(hd6)))  # [B, 64, 32, 32]
        # Conv layer:[B, 1024, 32, 32]
        hd4 = self.conv4d_2(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4, hd6_UT_hd4), 1)))  # [B, 512, 32, 32]

        # stage 3d
        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))  # [B, 32, 64, 64]
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))  # [B, 32, 64, 64]
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)  # [B, 128, 64, 64]
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4)  # [B, 256, 64, 64]
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(self.hd5DimreductSt3(hd5)))  # [B, 32, 64, 64]
        hd6_UT_hd3 = self.hd6_UT_hd3_conv(self.hd6_UT_hd3(self.hd6DimreductSt3(hd6)))  # [B, 32, 64, 64]
        # Conv layer:[B, 512, 64, 64]
        hd3 = self.conv3d_2(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3, hd6_UT_hd3), 1)))  # [B, 256, 64, 64]

        # stage 2d
        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))  # [B, 16, 128, 128]
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)  # [B, 64, 128, 128]
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3)  # [B, 128, 128, 128]
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(self.hd4DimreductSt2(hd4)))  # [B, 16, 128, 128]
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(self.hd5DimreductSt2(hd5)))  # [B, 16, 128, 128]
        hd6_UT_hd2 = self.hd6_UT_hd2_conv(self.hd6_UT_hd2(self.hd6DimreductSt2(hd6)))  # [B, 16, 128, 128]
        # Conv layer:[B, 256, 128, 128]
        hd2 = self.conv2d_2(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2, hd6_UT_hd2),
                      1)))  # [B, 128, 128, 128]

        # stage 1d
        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)  # [B, 32, 256, 256]
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2)  # [B, 64, 256, 256]
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(self.hd3DimreductSt1(hd3)))  # [B, 8, 256, 256]
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(self.hd4DimreductSt1(hd4)))  # [B, 8, 256, 256]
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(self.hd5DimreductSt1(hd5)))  # [B, 8, 256, 256]
        hd6_UT_hd1 = self.hd6_UT_hd1_conv(self.hd6_UT_hd1(self.hd6DimreductSt1(hd6)))  # [B, 8, 256, 256]
        # Conv layer:[B, 128, 256, 256]
        hd1 = self.conv1d_2(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1, hd6_UT_hd1),
                      1)))  # [B, 64, 256, 256]

        d1 = self.outconv1(hd1)  # [B, 2, 256, 256]
        return d1