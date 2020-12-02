# coding: utf-8

import math
import torch
from torch import nn

from .BasicModule import BasicModule


class SiameseNet(BasicModule):
    def __init__(self, num_classes, init_weights=True):
        super(SiameseNet, self).__init__()

        # 将Vgg16前三层的channel数除以16
        conv1 = [nn.Conv2d(1, 4, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(4, 4, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(8, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv3 = [nn.Conv2d(8, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        self.features = nn.Sequential(*(conv1 + conv2 + conv3))

        self.classifier = nn.Linear(in_features=3136, out_features=num_classes)
        # self.res_classifier = nn.Linear(in_features=3136, out_features=2)
        self.res_classifier = nn.Linear(in_features=6272, out_features=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        features_x = self.features(x)
        features_y = self.features(y)
        # print(features.size())
        # raise KeyboardInterrupt
        features_x = features_x.view(features_x.size(0), -1)
        features_y = features_y.view(features_y.size(0), -1)
        out_x = self.classifier(features_x)
        out_y = self.classifier(features_y)
        # out_res = self.res_classifier(torch.abs(features_x - features_y))
        out_res = self.res_classifier(torch.cat((features_x, features_y), dim=1))
        return out_x, out_y, out_res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
