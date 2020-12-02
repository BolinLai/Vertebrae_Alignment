# coding: utf-8

import os
import random
import numpy as np
import torch
import csv

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


class SlideWindowDataset(object):
    def __init__(self, csv_path, phase, useRGB=True, usetrans=True, balance=True):

        self.csv_path = csv_path
        self.phase = phase
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance

        self.images, self.labels = self.prepare_data()

        if self.usetrans:
            if self.phase == 'train':
                self.trans = transforms.Compose([
                    transforms.Resize((224, 224)),
                    # transforms.Resize((448, 448)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                    # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
                ])
            elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                self.trans = transforms.Compose([
                    transforms.Resize((224, 224)),
                    # transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                    # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
                ])
            else:
                raise IndexError
        else:
            self.trans = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.Resize((448, 448)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
                # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
            ])

    def __getitem__(self, index):
        image_path = self.images[index]
        # image_path = image_path.replace('SW_revise', 'TightMaskedImage_revise')
        image_path = image_path.replace('SW_revise', 'MaskedImage_revise')

        image = Image.open(image_path)
        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个
        image = self.trans(image)

        label = self.labels[index]

        return image, label, image_path

    # EdgeMask
    # def __getitem__(self, index):
    #     image_path = self.images[index]
    #     edge_mask_path = image_path.replace('SW_revise', 'EdgeMask_revise')
    #     image = Image.open(image_path)
    #     edge_mask = Image.open(edge_mask_path)
    #
    #     image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个
    #     edge_mask = Image.fromarray(np.asarray(edge_mask)[:, :, 0]) if not self.useRGB else edge_mask
    #
    #     image = functional.resize(image, (224, 224), Image.BILINEAR)
    #     edge_mask = functional.resize(edge_mask, (224, 224), Image.BILINEAR)
    #     # image = functional.resize(image, (448, 448), Image.BILINEAR)
    #     # edge_mask = functional.resize(edge_mask, (448, 448), Image.BILINEAR)
    #     if self.usetrans:
    #         if self.phase == 'train':
    #             if random.random() < 0.5:
    #                 image = functional.hflip(image)
    #                 edge_mask = functional.hflip(edge_mask)
    #             if random.random() < 0.5:
    #                 image = functional.vflip(image)
    #                 edge_mask = functional.vflip(edge_mask)
    #             angle = random.uniform(-30, 30)
    #             image = functional.rotate(image, angle)
    #             edge_mask = functional.rotate(edge_mask, angle)
    #         elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
    #             pass
    #         else:
    #             raise IndexError
    #     else:
    #         pass
    #
    #     image = functional.to_tensor(image)
    #     edge_mask = functional.to_tensor(edge_mask)  # 自动把mask变为0-1
    #
    #     label = self.labels[index]
    #
    #     return image, label, edge_mask, image_path

    def __len__(self):
        return len(self.images)

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()
        if self.phase == 'train' or self.phase == 'val':
            random.shuffle(lines)
        elif self.phase == 'test' or self.phase == 'test_train':
            lines.sort(key=lambda x: (x.split(',')[0].split('/')[-2], int(x.split(',')[0].split('/')[-1].split('_')[0][2:])))  # 按照病人id和滑窗id排序
        else:
            raise ValueError
        if self.balance:
            images, labels = [], []
            chosen_line = []
            normal_count = 0
            # threshold = 6800  # 使用三种程度的脱位 training data 取6800个无病的，validation data，test data 取所有的
            threshold = 6400  # 使用三种程度的脱位 training data 取6400个无病的，validation data，test data 取所有的
            for x in tqdm(lines, desc=f"Preparing balanced {self.phase} data"):
                image_path = str(x).strip().split(',')[0]
                label = int(str(x).strip().split(',')[1])
                if self.phase == 'train':
                    if label == 0 and normal_count < threshold:
                        images.append(image_path)
                        labels.append(label)
                        chosen_line.append((image_path, label))
                        normal_count += 1
                    elif label == 1:
                        images.append(image_path)
                        labels.append(label)
                        chosen_line.append((image_path, label))
                    else:
                        pass
                elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                    images.append(image_path)
                    labels.append(label)
                else:
                    raise ValueError
            if self.phase == 'train':
                chosen_line.sort(key=lambda y: (y[0].split('/')[-2], int(y[0].split('/')[-1].split('_')[0][2:])))
                with open(os.path.join('dataset', 'train_chosen_4.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(chosen_line)

        else:
            images = [str(x).strip().split(',')[0] for x in tqdm(lines, desc='Preparing Images')]
            labels = [int(str(x).strip().split(',')[1]) for x in tqdm(lines, desc='Preparing Labels')]
        return images, labels

    def dist(self):
        dist = {}
        for l in tqdm(self.labels, desc="Counting data distribution"):
            if str(l) in dist.keys():
                dist[str(l)] += 1
            else:
                dist[str(l)] = 1
        return dist


if __name__ == '__main__':
    train_data = SlideWindowDataset(csv_path=['dataset/train_D1F1.csv'], phase='train', useRGB=True, usetrans=True, balance=False)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    # val_data = SlideWindowDataset(csv_path=['dataset/test_s4v2.csv'], phase='val', useRGB=False, usetrans=True, balance=False)
    # val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
    # test_data = SlideWindowDataset(csv_path=['dataset/test_S4Shift_2010.csv'], phase='test', useRGB=False, usetrans=True, balance=False)
    # test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    # print(train_data.dist())

    # 查看做完transform之后的image
    # count = 0
    # for img, lab, segmask, img_path in tqdm(train_data):
    #     print(img.size(), segmask.size())
    #     img_1 = (img.numpy()*255)[0]
    #     img_2 = ((img * segmask).numpy()*255)[0]
    #     img_3 = (segmask.numpy()*255)[0]
    #
    #     im = Image.fromarray(img_1.astype('uint8'))
    #     im.save(os.path.join('/DB/rhome/bllai/Data/DATA5/Vertebrae/Sagittal/test', img_path.split('/')[-2] + "_" + 'img1' + "_" + img_path.split('/')[-1]))
    #     im = Image.fromarray(img_2.astype('uint8'))
    #     im.save(os.path.join('/DB/rhome/bllai/Data/DATA5/Vertebrae/Sagittal/test', img_path.split('/')[-2] + "_" + 'img2' + "_" + img_path.split('/')[-1]))
    #     im = Image.fromarray(img_3.astype('uint8'))
    #     im.save(os.path.join('/DB/rhome/bllai/Data/DATA5/Vertebrae/Sagittal/test', img_path.split('/')[-2] + "_" + 'img3' + "_" + img_path.split('/')[-1]))
    #     count += 1
    #     if count == 20:
    #         raise KeyboardInterrupt
    #     pass

    for img, lab, segmask, img_path in tqdm(train_data):
        print(segmask.size())
        print(segmask)
        print(torch.sum(segmask))
        raise KeyboardInterrupt
        # pass

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
