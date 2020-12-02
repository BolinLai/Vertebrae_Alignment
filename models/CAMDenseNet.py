# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import densenet121, densenet121_pre
from .BasicModule import BasicModule


class CAMDenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(CAMDenseNet121, self).__init__()

        self.conv0 = densenet121.features.conv0
        self.norm0 = densenet121.features.norm0
        self.relu0 = densenet121.features.relu0
        self.pool0 = densenet121.features.pool0
        self.denseblock1 = densenet121.features.denseblock1
        self.transition1 = densenet121.features.transition1
        self.denseblock2 = densenet121.features.denseblock2
        self.transition2 = densenet121.features.transition2
        self.denseblock3 = densenet121.features.denseblock3
        self.transition3 = densenet121.features.transition3
        self.denseblock4 = densenet121.features.denseblock4
        self.norm5 = densenet121.features.norm5

        self.avg_pool = densenet121.avg_pool
        self.classifier = nn.Linear(1024, num_classes, bias=True)

        self.CAMReLu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, calcam=True):
        feature = self.conv0(x)
        feature = self.norm0(feature)
        feature = self.relu0(feature)
        feature = self.pool0(feature)
        feature = self.denseblock1(feature)
        feature = self.transition1(feature)
        feature = self.denseblock2(feature)
        feature = self.transition2(feature)
        feature = self.denseblock3(feature)
        feature = self.transition3(feature)
        feature = self.denseblock4(feature)
        feature = self.norm5(feature)
        feature = functional.relu(feature, inplace=True)

        if not calcam:
            feature = self.avgpool(feature)
            feature = feature.view(feature.size(0), -1)
            out = self.fc(feature)
            return out
        else:
            weight_0 = self.classifier.weight[0].unsqueeze(1).unsqueeze(2)
            weight_1 = self.classifier.weight[1].unsqueeze(1).unsqueeze(2)

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
