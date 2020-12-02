# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .BasicModule import BasicModule
from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre


# class double_conv(nn.Module):
#     # (conv => BN => ReLU) * 2
#
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )
#
#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, conv, upsample):
#         super(up, self).__init__()
#
#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if upsample == 'bilinear':
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         elif upsample == 'transpose':
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
#
#         self.conv = conv
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
#
#         # for padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# class UNet(BasicModule):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride  # 没用

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


class doubleBasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(doubleBasicBlock, self).__init__()

        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.block1 = BasicBlock(inplanes=in_ch, planes=out_ch, stride=1, downsample=downsample)
        self.block2 = BasicBlock(inplanes=out_ch, planes=out_ch, stride=1, downsample=None)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, upsample):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsample == 'transpose':
            self.up = nn.ConvTranspose2d(in_ch1, in_ch1, 2, stride=2)

        self.conv = doubleBasicBlock(in_ch1 + in_ch2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet_Classifier(BasicModule):
    def __init__(self, num_classes):
        super(UNet_Classifier, self).__init__()

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        self.up1 = up(in_ch1=256, in_ch2=128, out_ch=64, upsample='transpose')
        self.up2 = up(in_ch1=64, in_ch2=64, out_ch=64, upsample='transpose')
        self.up3 = up(in_ch1=64, in_ch2=64, out_ch=64, upsample='transpose')

        self.transposeconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(in_channels=64, out_channels=num_classes-1, kernel_size=1, stride=1)

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_1 = self.bn1(feature_1)
        feature_1 = self.relu(feature_1)

        feature_2 = self.maxpool(feature_1)
        feature_2 = self.layer1(feature_2)

        feature_3 = self.layer2(feature_2)

        feature_4 = self.layer3(feature_3)

        # classification
        feature_5 = self.layer4(feature_4)

        feature = self.avgpool(feature_5)
        feature = feature.view(feature.size(0), -1)
        logits = self.fc(feature)

        # segmentation
        feature_3 = self.up1(feature_4, feature_3)
        feature_2 = self.up2(feature_3, feature_2)
        feature_1 = self.up3(feature_2, feature_1)

        feature = self.transposeconv(feature_1)
        logits_mask = self.outconv(feature)

        # CAM可视化s
        # weight_0 = self.fc.weight[0].unsqueeze(1).unsqueeze(2)
        # weight_1 = self.fc.weight[1].unsqueeze(1).unsqueeze(2)
        #
        # cam_0 = functional.relu(torch.sum(feature_5 * weight_0, dim=1, keepdim=True))
        # cam_1 = functional.relu(torch.sum(feature_5 * weight_1, dim=1, keepdim=True))
        #
        # cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        #
        # cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

        return logits, logits_mask
        # return logits, logits_mask, cam_0, cam_1


