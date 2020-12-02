# coding: utf-8

import torch
from torch import nn

from .pretrained import alexnet, alexnet_pre
from .BasicModule import BasicModule


class AlexNet(BasicModule):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = alexnet.features

        self.dropout1 = alexnet.classifier[0]
        self.fc1 = alexnet.classifier[1]
        self.relu1 = alexnet.classifier[2]
        self.dropout2 = alexnet.classifier[3]
        self.fc2 = alexnet.classifier[4]
        self.relu2 = alexnet.classifier[5]
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(features.size(0), -1)
        ref1 = self.fc1(self.dropout1(flat_features))
        ref2 = self.fc2(self.dropout2(self.relu1(ref1)))
        out = self.fc3(self.relu2(ref2))
        return out
