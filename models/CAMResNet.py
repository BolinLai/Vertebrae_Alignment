# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class CAMResNet18(BasicModule):
    def __init__(self, num_classes):
        super(CAMResNet18, self).__init__()

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        # self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc = nn.Linear(512, num_classes)

        self.CAMReLu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, calcam=True):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        if not calcam:
            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            out = self.fc(feature)
            return out
        else:
            weight_0 = self.fc.weight[0].unsqueeze(1).unsqueeze(2)
            weight_1 = self.fc.weight[1].unsqueeze(1).unsqueeze(2)

            cam_0 = self.CAMReLu(torch.sum(feature * weight_0, dim=1, keepdim=True))
            cam_1 = self.CAMReLu(torch.sum(feature * weight_1, dim=1, keepdim=True))

            cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

            cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

            # 防止在做loss的时候取对数时值为0
            cam_0 = torch.where(cam_0 > 0.001, cam_0, torch.Tensor([0.001]).cuda())
            cam_1 = torch.where(cam_1 > 0.001, cam_1, torch.Tensor([0.001]).cuda())

            # T = 0
            # cam_0 = torch.where(cam_0 < 0.001, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.001, cam_1, torch.Tensor([0.999]).cuda())

            # T = 0.25
            # cam_0 = torch.where(cam_0 < 0.25, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.25, cam_1, torch.Tensor([0.999]).cuda())

            # T = 0.5  # Best
            cam_0 = torch.where(cam_0 < 0.5, cam_0, torch.Tensor([0.999]).cuda())
            cam_1 = torch.where(cam_1 < 0.5, cam_1, torch.Tensor([0.999]).cuda())

            # T = 0.75
            # cam_0 = torch.where(cam_0 < 0.75, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.75, cam_1, torch.Tensor([0.999]).cuda())

            # T = 0.999(1.0) 相当于对边界上所有点都加约束
            # cam_0 = torch.where(cam_0 < 0.999, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.999, cam_1, torch.Tensor([0.999]).cuda())

            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            out = self.fc(feature)
            return out, cam_0, cam_1


class CAMResNet50(BasicModule):
    def __init__(self, num_classes):
        super(CAMResNet50, self).__init__()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        # self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        self.CAMReLu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, calcam=True):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        if not calcam:
            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            out = self.fc(feature)
            return out
        else:
            weight_0 = self.fc.weight[0].unsqueeze(1).unsqueeze(2)
            weight_1 = self.fc.weight[1].unsqueeze(1).unsqueeze(2)

            cam_0 = self.CAMReLu(torch.sum(feature * weight_0, dim=1, keepdim=True))
            cam_1 = self.CAMReLu(torch.sum(feature * weight_1, dim=1, keepdim=True))

            cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

            cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

            # 防止在做loss的时候取对数时值为0
            cam_0 = torch.where(cam_0 > 0.001, cam_0, torch.Tensor([0.001]).cuda())
            cam_1 = torch.where(cam_1 > 0.001, cam_1, torch.Tensor([0.001]).cuda())
            cam_0 = torch.where(cam_0 < 0.5, cam_0, torch.Tensor([0.999]).cuda())
            cam_1 = torch.where(cam_1 < 0.5, cam_1, torch.Tensor([0.999]).cuda())

            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            out = self.fc(feature)
            return out, cam_0, cam_1
