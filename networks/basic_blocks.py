import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=True, bias=False):
        super(ResidualBlock, self).__init__()
        dim_out = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes),
            )
        elif isinstance(downsample, nn.Module):
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock1DConv(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(BasicBlock1DConv, self).__init__()
        dim_out = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 9),
                                 stride=1, padding=(0, 4), bias=bias)
        self.conv2_2 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(9, 1),
                                 stride=1, padding=(4, 0), bias=bias)
        self.conv2_3 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(9, 1),
                                 stride=1, padding=(4, 0), bias=bias)
        self.conv2_4 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 9),
                                 stride=1, padding=(0, 4), bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x0 = self.conv2(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(self.h_transform(x))
        x3 = self.inv_h_transform(x3)
        x4 = self.conv2_4(self.v_transform(x))
        x4 = self.inv_v_transform(x4)

        x = x0 + torch.cat((x1, x2, x3, x4), 1)
        out = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DecoderBlock1DConv2(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock1DConv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv1 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x = torch.cat((x1, x2), 1)

        x = F.interpolate(x, scale_factor=2)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DecoderBlock1DConv4(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock1DConv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv1 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4)
        )

        self.norm2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 8)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 8, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = F.interpolate(x, scale_factor=2)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

