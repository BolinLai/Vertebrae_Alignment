# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import alexnet, alexnet_pre
from .BasicModule import BasicModule


class CAMAlexNet(BasicModule):
    def __init__(self, num_classes):
        super(CAMAlexNet, self).__init__()

        self.features = alexnet.features

        self.dropout1 = alexnet.classifier[0]
        self.fc1 = alexnet.classifier[1]
        self.relu1 = alexnet.classifier[2]
        self.dropout2 = alexnet.classifier[3]
        self.fc2 = alexnet.classifier[4]
        self.relu2 = alexnet.classifier[5]
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.GAP = nn.AvgPool2d(kernel_size=6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, calcam=True):
        # 计算得到预测的score
        features = self.features(x)
        flat_features = features.view(features.size(0), -1)
        ref1 = self.fc1(self.dropout1(flat_features))
        ref2 = self.fc2(self.dropout2(self.relu1(ref1)))
        out = self.fc3(self.relu2(ref2))

        if not calcam:
            return out
        else:
            # 计算CAM
            # 得到连续相乘的几个矩阵
            r1 = torch.diag_embed(torch.where(ref1 > 0, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda()))
            r2 = torch.diag_embed(torch.where(ref2 > 0, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda()))
            weight1 = self.fc1.weight
            weight2 = self.fc2.weight
            weight3 = self.fc3.weight

            # 对矩阵补充维度以便进行矩阵乘法
            weight1 = torch.cat([weight1.unsqueeze(0)] * x.size(0))
            weight2 = torch.cat([weight2.unsqueeze(0)] * x.size(0))
            weight3 = torch.cat([weight3.unsqueeze(0)] * x.size(0))

            # 计算输出对于flat_features的梯度
            chain_grad = torch.bmm(weight3, torch.bmm(r2, torch.bmm(weight2, torch.bmm(r1, weight1))))

            # 得到CAM并升采样
            channel_weight_0 = self.GAP(chain_grad[:, 0].view(features.size()))
            channel_weight_1 = self.GAP(chain_grad[:, 1].view(features.size()))

            cam_0 = self.relu(torch.sum(features * channel_weight_0, dim=1, keepdim=True))
            cam_1 = self.relu(torch.sum(features * channel_weight_1, dim=1, keepdim=True))

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
            return out, cam_0, cam_1
