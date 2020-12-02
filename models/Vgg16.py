# coding: utf-8

import torch
from torch import nn

from .pretrained import vgg16, vgg16_pre
from .BasicModule import BasicModule


class Vgg16(BasicModule):
    def __init__(self, num_classes):
        super(Vgg16, self).__init__()

        self.features = vgg16.features

        self.fc1 = vgg16.classifier[0]
        self.relu1 = vgg16.classifier[1]
        self.dropout1 = vgg16.classifier[2]
        self.fc2 = vgg16.classifier[3]
        self.relu2 = vgg16.classifier[4]
        self.dropout2 = vgg16.classifier[5]
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        # self.classifier = nn.Sequential(vgg16.classifier[0],
        #                                 vgg16.classifier[1],
        #                                 vgg16.classifier[2],
        #                                 vgg16.classifier[3],
        #                                 vgg16.classifier[4],
        #                                 vgg16.classifier[5],
        #                                 nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.fc1(features)
        out = self.fc2(self.dropout1(self.relu1(out)))
        out = self.fc3(self.dropout2(self.relu2(out)))
        # out = self.classifier(features)
        return out


class Modified_Vgg16(BasicModule):
    def __init__(self, num_classes):
        super(Modified_Vgg16, self).__init__()

        self.features = vgg16.features
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        feature = self.features(x)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out
