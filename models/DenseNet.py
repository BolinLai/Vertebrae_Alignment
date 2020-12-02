# coding: utf-8

import torch
from torch import nn
from torch.nn import functional

from .pretrained import densenet121, densenet121_pre
from .BasicModule import BasicModule


class DenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
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
        feature = self.avg_pool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.classifier(feature)

        return out
