# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import vgg16, vgg16_pre
from .BasicModule import BasicModule


class Modified_AGVgg16(BasicModule):
    def __init__(self, num_classes):
        super(Modified_AGVgg16, self).__init__()

        self.conv1 = nn.Sequential(vgg16.features[0], vgg16.features[1], vgg16.features[2], vgg16.features[3], vgg16.features[4])
        self.conv2 = nn.Sequential(vgg16.features[5], vgg16.features[6], vgg16.features[7], vgg16.features[8], vgg16.features[9])
        self.conv3 = nn.Sequential(vgg16.features[10], vgg16.features[11], vgg16.features[12], vgg16.features[13],
                                   vgg16.features[14], vgg16.features[15], vgg16.features[16])
        self.conv4 = nn.Sequential(vgg16.features[17], vgg16.features[18], vgg16.features[19], vgg16.features[20],
                                   vgg16.features[21], vgg16.features[22], vgg16.features[23])
        self.conv5 = nn.Sequential(vgg16.features[24], vgg16.features[25], vgg16.features[26], vgg16.features[27],
                                   vgg16.features[28], vgg16.features[29], vgg16.features[30])

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.AG1_conv1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG1_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG1_conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.AG1_relu = nn.ReLU(inplace=True)
        self.AG1_sigmoid = nn.Sigmoid()
        self.AG1_avgpool = nn.AvgPool2d(28, stride=1)

        self.AG2_conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG2_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG2_conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.AG2_relu = nn.ReLU(inplace=True)
        self.AG2_sigmoid = nn.Sigmoid()
        self.AG2_avgpool = nn.AvgPool2d(14, stride=1)

        self.fc = nn.Linear(256 + 512 + 512, num_classes)

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

    def forward(self, x, calcam=True):
        feature_1 = self.conv1(x)
        feature_2 = self.conv2(feature_1)
        feature_3 = self.conv3(feature_2)
        feature_4 = self.conv4(feature_3)
        feature_5 = self.conv5(feature_4)
        out = self.avgpool(feature_5)

        s1 = self.AG1_conv1(feature_3)
        g1 = self.AG1_conv2(feature_5)
        g1 = functional.interpolate(g1, size=(s1.size(2), s1.size(3)), mode='bilinear', align_corners=False)
        f1 = self.AG1_conv3(self.AG1_relu(s1 + g1))
        f1 = self.AG1_sigmoid(f1) * feature_3
        out_1 = self.AG1_avgpool(f1)

        s2 = self.AG2_conv1(feature_4)
        g2 = self.AG2_conv2(feature_5)
        g2 = functional.interpolate(g2, size=(s2.size(2), s2.size(3)), mode='bilinear', align_corners=False)
        f2 = self.AG2_conv3(self.AG2_relu(s2 + g2))
        f2 = self.AG2_sigmoid(f2) * feature_4
        out_2 = self.AG2_avgpool(f2)

        final_feature = torch.cat([out.view(out.size(0), -1), out_1.view(out_1.size(0), -1), out_2.view(out_2.size(0), -1)], 1)
        final_out = self.fc(final_feature)

        return final_out
