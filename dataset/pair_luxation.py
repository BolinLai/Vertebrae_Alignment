# coding: utf-8

import os
import random
import numpy as np
import torch

from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Size4 Train(AUG)
VERTEBRAE_MEAN = [0.342] * 3
VERTEBRAE_STD = [0.351] * 3


class PairSWDataset(object):
    def __init__(self, csv_path, phase, useRGB=True, usetrans=True, balance=True):

        self.csv_path = csv_path
        self.phase = phase
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance

        self.positive_images, self.positive_labels, self.negative_images, self.negative_labels = self.prepare_data()

        if self.usetrans:
            if self.phase == 'train':
                self.trans = transforms.Compose([
                    transforms.Resize((112, 112)),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    # transforms.Resize((224, 224)),
                    # transforms.Resize((56, 56)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                    # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
                ])
            elif self.phase in ['val_cls', 'val_pair', 'test']:
                self.trans = transforms.Compose([
                    # transforms.Resize((224, 224)),
                    transforms.Resize((112, 112)),
                    # transforms.Resize((56, 56)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                    # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
                ])
            else:
                raise ValueError
        else:
            self.trans = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((112, 112)),
                # transforms.Resize((56, 56)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
            ])

    def __getitem__(self, index):
        if self.phase in ('train', 'val_pair'):
            # 50%：正常-异常  25%：正常-正常  25%：异常-异常
            # a = random.randint(1, 100)
            # if 1 <= a <= 25:
            #     while True:
            #         index_1 = random.randint(0, len(self.positive_images) - 1)
            #         index_2 = random.randint(0, len(self.positive_images) - 1)
            #         if index_1 != index_2:
            #             break
            #     image_path_1 = self.positive_images[index_1]
            #     image_path_2 = self.positive_images[index_2]
            #     label_1, label_2, label_res = 1, 1, 0
            #
            # elif 26 <= a <= 50:
            #     while True:
            #         index_1 = random.randint(0, len(self.negative_images) - 1)
            #         index_2 = random.randint(0, len(self.negative_images) - 1)
            #         if index_1 != index_2:
            #             break
            #     image_path_1 = self.negative_images[index_1]
            #     image_path_2 = self.negative_images[index_2]
            #     label_1, label_2, label_res = 0, 0, 0
            #
            # elif 51 <= a <= 100:
            #     index_1 = random.randint(0, len(self.positive_images) - 1)
            #     index_2 = random.randint(0, len(self.negative_images) - 1)
            #     image_path_1 = self.positive_images[index_1]
            #     image_path_2 = self.negative_images[index_2]
            #     label_1, label_2, label_res = 1, 0, 1
            #
            # else:
            #     raise ValueError

            # 50%：正常-异常  50%：正常-正常
            a = random.randint(1, 100)
            if 1 <= a <= 50:
                while True:
                    index_1 = random.randint(0, len(self.negative_images) - 1)
                    index_2 = random.randint(0, len(self.negative_images) - 1)
                    if index_1 != index_2:
                        break
                image_path_1 = self.negative_images[index_1]
                image_path_2 = self.negative_images[index_2]
                label_1, label_2, label_res = 0, 0, 0

            elif 51 <= a <= 100:
                index_1 = random.randint(0, len(self.positive_images) - 1)
                index_2 = random.randint(0, len(self.negative_images) - 1)
                image_path_1 = self.positive_images[index_1]
                image_path_2 = self.negative_images[index_2]
                label_1, label_2, label_res = 1, 0, 1

            else:
                raise ValueError

        elif self.phase in ('val_cls', 'test'):
            image_path_1 = image_path_2 = (self.positive_images + self.negative_images)[index]
            label_1 = label_2 = (self.positive_labels + self.negative_labels)[index]
            label_res = 0

        else:
            raise ValueError

        image_1, image_2 = Image.open(image_path_1), Image.open(image_path_2)
        image_1 = Image.fromarray(np.asarray(image_1)[:, :, 0]) if not self.useRGB else image_1  # 得到的RGB图片三通道数值相等，只选择其中一个
        image_2 = Image.fromarray(np.asarray(image_2)[:, :, 0]) if not self.useRGB else image_2  # 得到的RGB图片三通道数值相等，只选择其中一个
        image_1 = self.trans(image_1)
        image_2 = self.trans(image_2)

        return image_1, image_2, label_1, label_2, label_res, image_path_1, image_path_2

    def __len__(self):
        if self.phase == 'train':
            length = 10000
        elif self.phase == 'val_pair':
            length = 2000
        elif self.phase == 'val_cls' or self.phase == 'test':
            length = len(self.positive_images + self.negative_images)
        # elif self.phase == 'test':
        #     length = len(self.image)
        else:
            raise ValueError
        return length

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()
        if self.phase == 'train' or self.phase == 'val_cls' or self.phase == 'val_pair':
            random.shuffle(lines)
        elif self.phase == 'test':
            lines.sort(key=lambda x: (x.split(',')[0].split('/')[-2], int(x.split(',')[0].split('/')[-1].split('_')[0][2:])))
        else:
            raise ValueError

        positive_images, positive_labels = [], []
        negative_images, negative_labels = [], []
        images, labels = [], []
        if self.balance:
            normal_count = 0
            threshold = 4200  # training data 取4200个无病的，validation data，test data 取所有的
            for x in tqdm(lines, desc=f"Preparing balanced {self.phase} data"):
                image_path = str(x).strip().split(',')[0]
                label = int(str(x).strip().split(',')[1])
                if self.phase == 'train':
                    if label == 0 and normal_count < threshold:
                        negative_images.append(image_path)
                        negative_labels.append(label)
                        normal_count += 1
                    elif label == 1:
                        positive_images.append(image_path)
                        positive_labels.append(label)
                    else:
                        pass
                elif self.phase in ('val_cls', 'val_pair', 'test'):
                    if label == 0:
                        negative_images.append(image_path)
                        negative_labels.append(label)
                    elif label == 1:
                        positive_images.append(image_path)
                #         positive_labels.append(label)
                # elif self.phase == 'test':
                #     images.append(image_path)
                #     labels.append(label)
                else:
                    raise ValueError
        else:
            for x in tqdm(lines, desc=f"Preparing data"):
                image_path = str(x).strip().split(',')[0]
                label = int(str(x).strip().split(',')[1])
                if label == 0:
                    negative_images.append(image_path)
                    negative_labels.append(label)
                elif label == 1:
                    positive_images.append(image_path)
                    positive_labels.append(label)

        assert len(positive_images) == len(positive_labels) and len(negative_images) == len(negative_labels)

        return positive_images, positive_labels, negative_images, negative_labels

    def dist(self):
        return {'0': len(self.negative_labels), '1': len(self.positive_labels)}


if __name__ == '__main__':
    # train_data = PairSWDataset(csv_path=['dataset/train_s4v2_shift.csv'], phase='train', useRGB=False, usetrans=True, balance=True)
    # train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    # valcls_data = PairSWDataset(csv_path=['dataset/test_s4v2_shift.csv'], phase='val_cls', useRGB=False, usetrans=True, balance=False)
    # valcls_dataloader = DataLoader(valcls_data, batch_size=16, shuffle=False, num_workers=4)
    valpair_data = PairSWDataset(csv_path=['dataset/test_s4v2_shift.csv'], phase='val_pair', useRGB=False, usetrans=True, balance=False)
    valpair_dataloader = DataLoader(valpair_data, batch_size=16, shuffle=False, num_workers=4)
    # test_data = PairSWDataset(csv_path=['dataset/test_s4v2.csv'], phase='test', useRGB=False, usetrans=True, balance=False)
    # test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    # print(train_data.dist())

    for img_1, img_2, lab_1, lab_2, lab_res, img_path_1, img_path_2 in tqdm(valpair_dataloader):
        # print(img_1.size())
        # print(img_2.size())
        # print(lab_1, lab_2, lab_res)
        # print(img_path_1)
        # print(img_path_2)
        # raise KeyboardInterrupt
        pass

    # mean, std = [], []
    # for _ in tqdm(range(10)):  # 由于训练集需要做数据增广，所以取10次的均值
    #     m = []
    #     for img, lab, img_path in tqdm(train_data, leave=False):
    #         m.append(img.squeeze())
    #     m = torch.cat(m, 0)
    #     mean.append(m.mean())
    #     std.append(m.std())
    #     # print(im.size())
    #     tqdm.write(f"{mean[-1]}, {std[-1]}")
    # print('mean:', mean)
    # print('std:', std)
    # print('average mean:', np.average(mean))
    # print('average std:', np.average(std))
