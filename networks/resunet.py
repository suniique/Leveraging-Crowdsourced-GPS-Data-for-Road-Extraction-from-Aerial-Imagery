import torch
from torchvision import models
from .resnet import *
from .basic_blocks import *


class ResUNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, encoder_1dconv=0,  decoder_1dconv=0):
        super(ResUNet34, self).__init__()
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

        self.center = nn.Conv2d(
            512, 512, kernel_size=3, dilation=1, padding=1)

        if decoder_1dconv == 0:
            self.decoder_ = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder_ = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
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
            x = self.firstconv(x)
            # x = self.firstconv(x.narrow(1, 3, self.num_channels))

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


class ResUnet(nn.Module):
    def __init__(self, num_channels=3):
        super(ResUnet, self).__init__()

        self.res_conv = ResidualBlock

        self.down1 = self.res_conv(num_channels, 64)
        self.down2 = self.res_conv(64, 128)
        self.down3 = self.res_conv(128, 256)
        self.down4 = self.res_conv(256, 512)

        self.bridge = self.conv_stage(512, 512)

        # self.up4 = self.conv_stage(1024, 512)
        # self.up3 = self.conv_stage(512, 256)
        # self.up2 = self.conv_stage(256, 128)
        # self.up1 = self.conv_stage(128, 64)

        self.up4 = self.res_conv(1024, 512)
        self.up3 = self.res_conv(512, 256)
        self.up2 = self.res_conv(256, 128)
        self.up1 = self.res_conv(128, 64)

        self.trans4 = self.upsample(512, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            # nn.LeakyReLU(0.1),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            # nn.LeakyReLU(0.1),
            nn.ReLU(),
        )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))    # ch = 512

        out = self.bridge(self.max_pool(conv4_out))         # ch = 512

        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))
        out = self.conv_last(out)

        return out


class ResUnet1DConv(nn.Module):
    def __init__(self, num_channels=3):
        super(ResUnet1DConv, self).__init__()

        self.res_conv = ResidualBlock
        self.upsample = DecoderBlock1DConv4

        self.down1 = self.res_conv(num_channels, 64)
        self.down2 = self.res_conv(64, 128)
        self.down3 = self.res_conv(128, 256)
        self.down4 = self.res_conv(256, 512)

        self.bridge = self.conv_stage(512, 512)

        self.up4 = self.res_conv(1024, 512)
        self.up3 = self.res_conv(512, 256)
        self.up2 = self.res_conv(256, 128)
        self.up1 = self.res_conv(128, 64)

        self.trans4 = self.upsample(512, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))    # ch = 512

        out = self.bridge(self.max_pool(conv4_out))         # ch = 512

        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))
        out = self.conv_last(out)

        return out
