# coding=utf-8

import torch
from torch.utils.data import DataLoader
from pprint import pprint

from config import config
from dataset import SlideWindowDataset
from models import UNet_Classifier, ResNet18

train_data = SlideWindowDataset(config.train_paths, phase='train', useRGB=config.useRGB, usetrans=config.usetrans, balance=config.data_balance)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=config.num_workers)

pprint(list(train_dataloader)[:10])
pprint(list(train_dataloader)[:10])
