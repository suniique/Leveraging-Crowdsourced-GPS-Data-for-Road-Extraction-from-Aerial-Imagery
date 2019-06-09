import torch
from torchvision import models
from .resnet import *
from .basic_blocks import *


class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, encoder_1dconv=0,  decoder_1dconv=0):
        super(DinkNet34, self).__init__()
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)

        self.dblock = DBlock(512)

        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4

        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        # Encoder
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add
        else:
            x = self.firstconv(x)

        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)



class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, decoder_1dconv=0, using_resnet=True):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        self.num_channels = num_channels

        if using_resnet:
            resnet = models.resnet34(pretrained=True)
            if self.num_channels > 3:
                self.addconv = nn.Conv2d(
                    self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)
                self.firstconv = resnet.conv1
            elif self.num_channels < 3:
                self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            else:
                self.firstconv = resnet.conv1
            self.firstbn = resnet.bn1
            self.firstrelu = resnet.relu
            self.firstmaxpool = resnet.maxpool
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            self.encoder1 = Encoder(64, 64, 3, 1, 1)
            self.encoder2 = Encoder(64, 128, 3, 2, 1)
            self.encoder3 = Encoder(128, 256, 3, 2, 1)
            self.encoder4 = Encoder(256, 512, 3, 2, 1)

        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4

        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add
        else:
            x = self.firstconv(x)

        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


