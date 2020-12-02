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


class SegLineDataset(object):
    def __init__(self, csv_path, phase, useRGB=True, usetrans=True, balance=True):
        self.csv_path = csv_path
        self.phase = phase
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance

        self.images, self.labels = self.prepare_data()

        self.image_trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        self.label_trans = transforms.Compose([
            transforms.Resize((112, 112)),
            # transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_path = self.images[index]
        line_mask_path = image_path.replace('SW_S4Shift_2010', 'LM_S4Shift_2010_2nd')
        image = Image.open(image_path)
        line_mask = Image.open(line_mask_path)

        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个

        if self.usetrans:
            if self.phase == 'train':
                if random.random() < 0.5:
                    image = functional.hflip(image)
                    line_mask = functional.hflip(line_mask)
                if random.random() < 0.5:
                    image = functional.vflip(image)
                    line_mask = functional.vflip(line_mask)
                angle = random.uniform(-30, 30)
                image = functional.rotate(image, angle)
                line_mask = functional.rotate(line_mask, angle)
            elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                pass
            else:
                raise IndexError
        else:
            pass

        image = self.image_trans(image)
        line_mask = self.label_trans(line_mask)
        line_mask = np.asarray(line_mask)

        label_mask = np.zeros(line_mask.shape, dtype=np.int16)
        label_mask[np.where(line_mask >= 240)] = 1
        label_mask = label_mask.astype(int)

        label_mask = torch.from_numpy(label_mask).long()

        label = self.labels[index]

        return image, label, label_mask, image_path

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
            lines.sort(key=lambda x: (x.split(',')[0].split('/')[-2], int(x.split(',')[0].split('/')[-1].split('_')[0][2:])))
        else:
            raise ValueError
        if self.balance:
            images, labels = [], []
            normal_count = 0
            # threshold = 4200  # training data 取4200个无病的，validation data，test data 取所有的
            threshold = 6800  # training data 取6800个无病的，validation data，test data 取所有的
            for x in tqdm(lines, desc=f"Preparing balanced {self.phase} data"):
                image_path = str(x).strip().split(',')[0]
                label = int(str(x).strip().split(',')[1])
                if self.phase == 'train':
                    if label == 0 and normal_count < threshold:
                        images.append(image_path)
                        labels.append(label)
                        normal_count += 1
                    elif label == 1:
                        images.append(image_path)
                        labels.append(label)
                    else:
                        pass
                elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                    images.append(image_path)
                    labels.append(label)
                else:
                    raise ValueError
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
    train_data = SegLineDataset(csv_path=['dataset/train_S4Shift_2010.csv'], phase='train', useRGB=False, usetrans=True, balance=False)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    count = 0
    for img, lab, lbm, img_path in tqdm(train_data):
        img = (img.numpy() * 255)[0]
        lbm = lbm.numpy() * 255

        im = Image.fromarray(lbm.astype('uint8'))
        im.save(os.path.join('/DB/rhome/bllai/Data/DATA5/Vertebrae/Sagittal/test_2', img_path.split('/')[-2] + "_lbl_" + img_path.split('/')[-1]))
        im = Image.fromarray(img.astype('uint8'))
        im.save(os.path.join('/DB/rhome/bllai/Data/DATA5/Vertebrae/Sagittal/test_2', img_path.split('/')[-2] + "_img_" + img_path.split('/')[-1]))

        count += 1
        if count == 20:
            raise KeyboardInterrupt
        pass
