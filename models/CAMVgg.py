# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .pretrained import vgg16, vgg16_pre
from .BasicModule import BasicModule


class CAMVgg16(BasicModule):
    def __init__(self, num_classes):
        super(CAMVgg16, self).__init__()

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

        self.GAP = nn.AvgPool2d(kernel_size=7)
        self.relu = nn.ReLU(inplace=True)

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
        # self.fc1 = self.classifier[0]
        # self.relu1 = self.classifier[1]
        # self.dropout1 = self.classifier[2]
        # self.fc2 = self.classifier[3]
        # self.relu2 = self.classifier[4]
        # self.dropout2 = self.classifier[5]
        # self.fc3 = self.classifier[6]

        # 计算得到预测的score
        features = self.features(x)
        flat_features = features.view(features.size(0), -1)
        ref1 = self.fc1(flat_features)
        ref2 = self.fc2(self.dropout1(self.relu1(ref1)))
        out = self.fc3(self.dropout2(self.relu2(ref2)))

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
            # cam_0 = torch.where(cam_0 > 0.001, cam_0, torch.Tensor([0.001]).cuda())
            # cam_1 = torch.where(cam_1 > 0.001, cam_1, torch.Tensor([0.001]).cuda())
            # cam_0 = torch.where(cam_0 < 0.5, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.5, cam_1, torch.Tensor([0.999]).cuda())
            return out, cam_0, cam_1


class Modified_CAMVgg16(BasicModule):
    def __init__(self, num_classes):
        super(Modified_CAMVgg16, self).__init__()

        self.features = vgg16.features
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.CAMReLu = nn.ReLU(inplace=True)

    def forward(self, x, calcam=True):
        feature = self.features(x)

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
