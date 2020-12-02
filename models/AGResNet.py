# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class AGResNet18(BasicModule):
    def __init__(self, num_classes):
        super(AGResNet18, self).__init__()

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool

        self.AG1_conv1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG1_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG1_conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.AG1_relu = nn.ReLU(inplace=True)
        self.AG1_sigmoid = nn.Sigmoid()
        self.AG1_avgpool = nn.AvgPool2d(28, stride=1)

        self.AG2_conv1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG2_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.AG2_conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.AG2_relu = nn.ReLU(inplace=True)
        self.AG2_sigmoid = nn.Sigmoid()
        self.AG2_avgpool = nn.AvgPool2d(14, stride=1)

        self.fc = nn.Linear(128 + 256 + 512, num_classes)

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

        feature_1 = self.layer1(feature)
        feature_2 = self.layer2(feature_1)
        feature_3 = self.layer3(feature_2)
        feature_4 = self.layer4(feature_3)
        out = self.avgpool(feature_4)

        s1 = self.AG1_conv1(feature_2)
        g1 = self.AG1_conv2(feature_4)
        g1 = functional.interpolate(g1, size=(s1.size(2), s1.size(3)), mode='bilinear', align_corners=False)
        f1 = self.AG1_conv3(self.AG1_relu(s1 + g1))
        f1 = self.AG1_sigmoid(f1) * feature_2
        out_1 = self.AG1_avgpool(f1)

        s2 = self.AG2_conv1(feature_3)
        g2 = self.AG2_conv2(feature_4)
        g2 = functional.interpolate(g2, size=(s2.size(2), s2.size(3)), mode='bilinear', align_corners=False)
        f2 = self.AG2_conv3(self.AG2_relu(s2 + g2))
        f2 = self.AG2_sigmoid(f2) * feature_3
        out_2 = self.AG2_avgpool(f2)

        final_feature = torch.cat([out.view(out.size(0), -1), out_1.view(out_1.size(0), -1), out_2.view(out_2.size(0), -1)], 1)
        final_out = self.fc(final_feature)

        # attmap_1 = functional.interpolate(f1, size=(224, 224), mode='bilinear', align_corners=False)
        # attmap_2 = functional.interpolate(f2, size=(224, 224), mode='bilinear', align_corners=False)
        # attmap_1 = attmap_1 - torch.min(torch.min(attmap_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # attmap_1 = attmap_1 / torch.max(torch.max(attmap_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # attmap_2 = attmap_2 - torch.min(torch.min(attmap_2, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # attmap_2 = attmap_2 / torch.max(torch.max(attmap_2, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

        # CAM可视化
        # weight_0 = self.fc.weight[0][384:].unsqueeze(1).unsqueeze(2)
        # weight_1 = self.fc.weight[1][384:].unsqueeze(1).unsqueeze(2)
        #
        # cam_0 = functional.relu(torch.sum(feature_4 * weight_0, dim=1, keepdim=True))
        # cam_1 = functional.relu(torch.sum(feature_4 * weight_1, dim=1, keepdim=True))
        #
        # cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        #
        # cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

        return final_out
        # return final_out, cam_0, cam_1
        # return final_out, attmap_1, attmap_2
