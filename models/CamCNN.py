# coding: utf-8

import math
import torch
from torch import nn
from torch.nn import functional

from .BasicModule import BasicModule


class CAM_CNN(BasicModule):
    def __init__(self, num_classes, init_weights=True):
        super(CAM_CNN, self).__init__()

        # Vgg16
        # conv1 = [nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv2 = [nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv3 = [nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv4 = [nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv5 = [nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # self.features = nn.Sequential(*(conv1 + conv2 + conv3 + conv4 + conv5))
        #
        # self.classifier = nn.Linear(in_features=25088, out_features=num_classes)
        #
        # self.GAP = nn.AvgPool2d(kernel_size=7)
        # self.relu = nn.ReLU(inplace=True)

        # 将Vgg16前4层的channel数除以8
        conv1 = [nn.Conv2d(1, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(8, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv2 = [nn.Conv2d(8, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv3 = [nn.Conv2d(16, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv4 = [nn.Conv2d(32, 64, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        self.features = nn.Sequential(*(conv1 + conv2 + conv3 + conv4))

        self.classifier = nn.Linear(in_features=12544, out_features=num_classes)

        self.GAP = nn.AvgPool2d(kernel_size=14)
        self.relu = nn.ReLU(inplace=True)

        # 将Vgg16前4层的channel数除以16
        # conv1 = [nn.Conv2d(1, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(4, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv3 = [nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv4 = [nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # self.features = nn.Sequential(*(conv1 + conv2 + conv3 + conv4))
        #
        # self.classifier = nn.Linear(in_features=6272, out_features=num_classes)
        #
        # self.GAP = nn.AvgPool2d(kernel_size=14)
        # self.relu = nn.ReLU(inplace=True)

        # 将Vgg16前3层的channel数除以16
        # conv1 = [nn.Conv2d(1, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(4, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv3 = [nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # self.features = nn.Sequential(*(conv1 + conv2 + conv3))
        #
        # self.classifier = nn.Linear(in_features=3136, out_features=num_classes)
        #
        # self.GAP = nn.AvgPool2d(kernel_size=14)
        # self.relu = nn.ReLU(inplace=True)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, calcam=True):
        features = self.features(x)

        if not calcam:
            features = features.view(features.size(0), -1)
            out = self.classifier(features)
            return out

        else:
            weight_0 = self.GAP(self.classifier.weight[0].view(features.size()[1:]))
            weight_1 = self.GAP(self.classifier.weight[1].view(features.size()[1:]))
            cam_0 = self.relu(torch.sum(features * weight_0, dim=1, keepdim=True))
            cam_1 = self.relu(torch.sum(features * weight_1, dim=1, keepdim=True))

            cam_0 = functional.interpolate(cam_0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            cam_1 = functional.interpolate(cam_1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

            cam_0 = cam_0 - torch.min(torch.min(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_0 = cam_0 / torch.max(torch.max(cam_0, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 - torch.min(torch.min(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            cam_1 = cam_1 / torch.max(torch.max(cam_1, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

            # cam_0 = torch.where(cam_0 < 0.3, cam_0, torch.Tensor([1.0]).cuda())
            # cam_1 = torch.where(cam_1 < 0.3, cam_1, torch.Tensor([1.0]).cuda())
            # cam_0 = torch.where(cam_0 < 0.5, cam_0, torch.Tensor([1.0]).cuda())
            # cam_1 = torch.where(cam_1 < 0.5, cam_1, torch.Tensor([1.0]).cuda())
            # cam_0 = torch.where(cam_0 < 0.7, cam_0, torch.Tensor([1.0]).cuda())
            # cam_1 = torch.where(cam_1 < 0.7, cam_1, torch.Tensor([1.0]).cuda())

            # 防止在做loss的时候取对数时值为0
            cam_0 = torch.where(cam_0 > 0.001, cam_0, torch.Tensor([0.001]).cuda())
            cam_1 = torch.where(cam_1 > 0.001, cam_1, torch.Tensor([0.001]).cuda())
            # cam_0 = torch.where(cam_0 < 0.999, cam_0, torch.Tensor([0.999]).cuda())
            # cam_1 = torch.where(cam_1 < 0.999, cam_1, torch.Tensor([0.999]).cuda())
            cam_0 = torch.where(cam_0 < 0.5, cam_0, torch.Tensor([0.999]).cuda())
            cam_1 = torch.where(cam_1 < 0.5, cam_1, torch.Tensor([0.999]).cuda())

            features = features.view(features.size(0), -1)
            out = self.classifier(features)
            return out, cam_0, cam_1

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


