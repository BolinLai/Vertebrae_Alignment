# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class PCResNet18(BasicModule):
    def __init__(self, num_classes):
        super(PCResNet18, self).__init__()

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.maxpool(fx)

        fx = self.layer1(fx)
        fx = self.layer2(fx)
        fx = self.layer3(fx)
        fx = self.layer4(fx)

        fy = self.conv1(y)
        fy = self.bn1(fy)
        fy = self.relu(fy)
        fy = self.maxpool(fy)

        fy = self.layer1(fy)
        fy = self.layer2(fy)
        fy = self.layer3(fy)
        fy = self.layer4(fy)

        # CAM可视化
        # weight_0 = self.fc.weight[0].unsqueeze(1).unsqueeze(2)
        # weight_1 = self.fc.weight[1].unsqueeze(1).unsqueeze(2)
        #
        # cam_0 = functional.relu(torch.sum(fx * weight_0, dim=1, keepdim=True))
        # cam_1 = functional.relu(torch.sum(fx * weight_1, dim=1, keepdim=True))
        #
        # cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        #
        # cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        # cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

        fx = self.avgpool(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc(fx)

        fy = self.avgpool(fy)
        fy = fy.view(fy.size(0), -1)
        out_y = self.fc(fy)

        return out_x, out_y, fx, fy
        # return out_x, out_y, fx, fy, cam_0, cam_1


class PCResNet50(BasicModule):
    def __init__(self, num_classes):
        super(PCResNet50, self).__init__()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.maxpool(fx)

        fx = self.layer1(fx)
        fx = self.layer2(fx)
        fx = self.layer3(fx)
        fx = self.layer4(fx)

        fx = self.avgpool(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc(fx)

        fy = self.conv1(y)
        fy = self.bn1(fy)
        fy = self.relu(fy)
        fy = self.maxpool(fy)

        fy = self.layer1(fy)
        fy = self.layer2(fy)
        fy = self.layer3(fy)
        fy = self.layer4(fy)

        fy = self.avgpool(fy)
        fy = fy.view(fy.size(0), -1)

        out_y = self.fc(fy)
        return out_x, out_y, fx, fy
