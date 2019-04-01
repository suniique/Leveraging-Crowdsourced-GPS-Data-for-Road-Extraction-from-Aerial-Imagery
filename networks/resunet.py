import torch
from torchvision import models
from .resnet import *
from .basic_blocks import *


class ResUNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, dir=False, encoder_1dconv=False, skip_dblock=False):
        super(ResUNet34, self).__init__()

        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        self.skip_dblock = skip_dblock

        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        if encoder_1dconv == False:
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

        self.center = nn.Conv2d(
            512, 512, kernel_size=3, dilation=1, padding=1)

        if dir == 0:
            self.decoder_ = DecoderBlock
        elif dir == 2:
            self.decoder_ = DecoderBlock1DConv2
        elif dir == 4:
            self.decoder_ = DecoderBlock1DConv4

        self.decoder4 = self.decoder_(filters[3], filters[2])
        self.decoder3 = self.decoder_(filters[2], filters[1])
        self.decoder2 = self.decoder_(filters[1], filters[0])
        self.decoder1 = self.decoder_(filters[0], filters[0])

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
        elif self.num_channels == 3:
            x = self.firstconv(x)
        else:
            x = self.firstconv(x.narrow(1, 3, self.num_channels))

        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.center(e4)

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
