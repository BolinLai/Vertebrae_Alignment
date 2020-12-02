# coding: utf-8
import os
import fire
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
from torchnet import meter

from config import config
from dataset import PairSWDataset
from models import SiameseNet
from utils import Visualizer, write_csv


def train(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)
    vis.log('Use config:')
    for k, v in config.__class__.__dict__.items():
        if not k.startswith('__'):
            vis.log(f"{k}: {getattr(config, k)}")

    # prepare data
    train_data = PairSWDataset(config.train_paths, phase='train', useRGB=config.useRGB, usetrans=config.usetrans, balance=config.data_balance)
    valpair_data = PairSWDataset(config.test_paths, phase='val_pair', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    valcls_data = PairSWDataset(config.test_paths, phase='val_cls', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    print('Training Samples:', train_data.__len__(), 'ValPair Samples:', valpair_data.__len__(), 'ValCls Samples:', valcls_data.__len__())
    dist = train_data.dist()
    print('Train Data Distribution:', dist)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valpair_dataloader = DataLoader(valpair_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    valcls_dataloader = DataLoader(valcls_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # prepare model
    model = SiameseNet(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])

    model.train()

    # criterion and optimizer
    weight_cls = torch.Tensor([1, 2])
    weight_pair = torch.Tensor([1, 1])
    vis.log(f'class loss weight: {weight_cls}')
    vis.log(f'pair loss weight: {weight_pair}')
    print('class loss weight:', weight_cls)
    print('pair loss weight:', weight_pair)
    weight_cls = weight_cls.cuda()
    weight_pair = weight_pair.cuda()
    cls_criterion = torch.nn.CrossEntropyLoss(weight=weight_cls)
    pair_criterion = torch.nn.CrossEntropyLoss(weight=weight_pair)

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # metric
    softmax = functional.softmax
    cls_loss_meter = meter.AverageValueMeter()
    cls_epoch_loss = meter.AverageValueMeter()
    pair_loss_meter = meter.AverageValueMeter()
    pair_epoch_loss = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    epoch_loss = meter.AverageValueMeter()

    cls_train_cm = meter.ConfusionMeter(config.num_classes)
    pair_train_cm = meter.ConfusionMeter(config.num_classes)
    # previous_loss = 100
    cls_previous_avg_se = 0

    # train
    if config.parallel:
        if not os.path.exists(os.path.join('checkpoints', model.module.model_name)):
            os.mkdir(os.path.join('checkpoints', model.module.model_name))
    else:
        if not os.path.exists(os.path.join('checkpoints', model.model_name)):
            os.mkdir(os.path.join('checkpoints', model.model_name))

    for epoch in range(config.max_epoch):
        print(f"epoch: [{epoch+1}/{config.max_epoch}] =============================================")
        cls_train_cm.reset()
        pair_train_cm.reset()
        cls_epoch_loss.reset()
        pair_epoch_loss.reset()

        # train
        for i, (image_1, image_2, label_1, label_2, label_res, _, _) in tqdm(enumerate(train_dataloader)):
            cls_loss_meter.reset()
            pair_loss_meter.reset()

            # prepare input
            image_1 = Variable(image_1)
            image_2 = Variable(image_2)
            target_1 = Variable(label_1)
            target_2 = Variable(label_2)
            target_res = Variable(label_res)

            if config.use_gpu:
                image_1 = image_1.cuda()
                image_2 = image_2.cuda()
                target_1 = target_1.cuda()
                target_2 = target_2.cuda()
                target_res = target_res.cuda()

            # go through the model
            score_1, score_2, score_res = model(image_1, image_2)

            # backpropagate
            optimizer.zero_grad()
            cls_loss = cls_criterion(score_1, target_1) + cls_criterion(score_2, target_2)
            pair_loss = pair_criterion(score_res, target_res)
            loss = cls_loss + pair_loss
            loss.backward()
            optimizer.step()

            cls_loss_meter.add(cls_loss.data[0])
            pair_loss_meter.add(pair_loss.data[0])
            cls_epoch_loss.add(cls_loss.data[0])
            pair_epoch_loss.add(pair_loss.data[0])
            loss_meter.add(loss.data[0])
            epoch_loss.add(loss.data[0])

            cls_train_cm.add(softmax(score_1, dim=1).data, target_1.data)
            cls_train_cm.add(softmax(score_2, dim=1).data, target_2.data)
            pair_train_cm.add(softmax(score_res, dim=1).data, target_res.data)

            if (i+1) % config.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                # print('loss', loss_meter.value()[0])

        # print result
        cls_train_se = [100. * cls_train_cm.value()[0][0] / (cls_train_cm.value()[0][0] + cls_train_cm.value()[0][1]),
                        100. * cls_train_cm.value()[1][1] / (cls_train_cm.value()[1][0] + cls_train_cm.value()[1][1])]
        pair_train_se = [100. * pair_train_cm.value()[0][0] / (pair_train_cm.value()[0][0] + pair_train_cm.value()[0][1]),
                         100. * pair_train_cm.value()[1][1] / (pair_train_cm.value()[1][0] + pair_train_cm.value()[1][1])]
        model.eval()
        cls_val_cm, cls_val_accuracy, cls_val_se = val_cls(model, valcls_dataloader)
        pair_val_cm, pair_val_accuracy, pair_val_se = val_pair(model, valpair_dataloader)

        if np.average(cls_val_se) > cls_previous_avg_se:  # 当测试集上的平均sensitivity升高时保存模型
            if config.parallel:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
                save_model_name = config.save_model_name if config.save_model_name else model.module.model_name + '_best_model.pth'
                if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                    os.makedirs(os.path.join('checkpoints', save_model_dir))
                model.module.save(os.path.join('checkpoints', save_model_dir, save_model_name))
            else:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
                save_model_name = config.save_model_name if config.save_model_name else model.model_name + '_best_model.pth'
                if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                    os.makedirs(os.path.join('checkpoints', save_model_dir))
                model.save(os.path.join('checkpoints', save_model_dir, save_model_name))
            cls_previous_avg_se = np.average(cls_val_se)

        if epoch+1 == config.max_epoch:  # 保存最后一个模型
            if config.parallel:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
                save_model_name = config.save_model_name.split('.pth')[0]+'_last.pth' if config.save_model_name else model.module.model_name + '_last_model.pth'
            else:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
                save_model_name = config.save_model_name.split('.pth')[0]+'_last.pth' if config.save_model_name else model.model_name + '_last_model.pth'
            if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                os.makedirs(os.path.join('checkpoints', save_model_dir))
            model.save(os.path.join('checkpoints', save_model_dir, save_model_name))

        vis.plot_many({'cls_epoch_loss': cls_epoch_loss.value()[0], 'pair_epoch_loss': pair_epoch_loss.value()[0], 'epoch_loss': epoch_loss.value()[0],
                       'cls_train_avg_se': np.average(cls_train_se), 'cls_train_se_0': cls_train_se[0], 'cls_train_se_1': cls_train_se[1],
                       'pair_train_avg_se': np.average(pair_train_se), 'pair_train_se_0': pair_train_se[0], 'pair_train_se_1': pair_train_se[1],
                       'cls_val_avg_se': np.average(cls_val_se), 'cls_val_se_0': cls_val_se[0], 'cls_val_se_1': cls_val_se[1],
                       'pair_val_avg_se': np.average(pair_val_se), 'pair_val_se_0': pair_val_se[0], 'pair_val_se_1': pair_val_se[1]})
        vis.log(f"epoch: [{epoch+1}/{config.max_epoch}] ===============================================")
        vis.log(f"lr: {lr}, loss: {round(epoch_loss.value()[0], 5)}")
        vis.log(f"cls_train_avg_se: {round(np.average(cls_train_se), 4)}, cls_train_se_0: {round(cls_train_se[0], 4)}, cls_train_se_1: {round(cls_train_se[1], 4)}")
        vis.log(f"pair_train_avg_se: {round(np.average(pair_train_se), 4)}, pair_train_se_0: {round(pair_train_se[0], 4)}, pair_train_se_1: {round(pair_train_se[1], 4)}")
        vis.log(f"cls_val_avg_se: {round(sum(cls_val_se)/len(cls_val_se), 4)}, cls_val_se_0: {round(cls_val_se[0], 4)}, cls_val_se_1: {round(cls_val_se[1], 4)}")
        vis.log(f"pair_val_avg_se: {round(sum(pair_val_se) / len(pair_val_se), 4)}, pair_val_se_0: {round(pair_val_se[0], 4)}, pair_val_se_1: {round(pair_val_se[1], 4)}")
        vis.log(f'cls_train_cm: {cls_train_cm.value()}')
        vis.log(f'pair_train_cm: {pair_train_cm.value()}')
        vis.log(f'cls_val_cm: {cls_val_cm.value()}')
        vis.log(f'pair_val_cm: {pair_val_cm.value()}')
        print("lr:", lr, "loss:", round(epoch_loss.value()[0], 5))
        print('cls_train_avg_se:', round(np.average(cls_train_se), 4), 'cls_train_se_0:', round(cls_train_se[0], 4), 'cls_train_se_1:', round(cls_train_se[1], 4))
        print('pair_train_avg_se:', round(np.average(pair_train_se), 4), 'pair_train_se_0:', round(pair_train_se[0], 4), 'pair_train_se_1:', round(pair_train_se[1], 4))
        print('cls_val_avg_se:', round(np.average(cls_val_se), 4), 'cls_val_se_0:', round(cls_val_se[0], 4), 'cls_val_se_1:', round(cls_val_se[1], 4))
        print('pair_val_avg_se:', round(np.average(pair_val_se), 4), 'pair_val_se_0:', round(pair_val_se[0], 4), 'pair_val_se_1:', round(pair_val_se[1], 4))
        print('cls_train_cm:')
        print(cls_train_cm.value())
        print('pair_train_cm:')
        print(pair_train_cm.value())
        print('cls_val_cm:')
        print(cls_val_cm.value())
        print('pair_val_cm:')
        print(pair_val_cm.value())

        # update learning rate
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * config.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # previous_loss = loss_meter.value()[0]
        if (epoch+1) % 5 == 0:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def train_pair(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)
    vis.log('Use config:')
    for k, v in config.__class__.__dict__.items():
        if not k.startswith('__'):
            vis.log(f"{k}: {getattr(config, k)}")

    # prepare data
    train_data = PairSWDataset(config.train_paths, phase='train', useRGB=config.useRGB, usetrans=config.usetrans, balance=config.data_balance)
    valpair_data = PairSWDataset(config.test_paths, phase='val_pair', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    print('Training Samples:', train_data.__len__(), 'ValPair Samples:', valpair_data.__len__())
    dist = train_data.dist()
    print('Train Data Distribution:', dist)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valpair_dataloader = DataLoader(valpair_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # prepare model
    model = SiameseNet(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])

    model.train()

    # criterion and optimizer
    weight_pair = torch.Tensor([1, 1.5])
    vis.log(f'pair loss weight: {weight_pair}')
    print('pair loss weight:', weight_pair)
    weight_pair = weight_pair.cuda()
    pair_criterion = torch.nn.CrossEntropyLoss(weight=weight_pair)

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # metric
    softmax = functional.softmax
    pair_loss_meter = meter.AverageValueMeter()
    pair_epoch_loss = meter.AverageValueMeter()

    pair_train_cm = meter.ConfusionMeter(config.num_classes)
    # previous_loss = 100
    pair_previous_avg_se = 0

    # train
    if config.parallel:
        if not os.path.exists(os.path.join('checkpoints', model.module.model_name)):
            os.mkdir(os.path.join('checkpoints', model.module.model_name))
    else:
        if not os.path.exists(os.path.join('checkpoints', model.model_name)):
            os.mkdir(os.path.join('checkpoints', model.model_name))

    for epoch in range(config.max_epoch):
        print(f"epoch: [{epoch+1}/{config.max_epoch}] =============================================")
        pair_train_cm.reset()
        pair_epoch_loss.reset()

        # train
        for i, (image_1, image_2, label_1, label_2, label_res, _, _) in tqdm(enumerate(train_dataloader)):
            pair_loss_meter.reset()

            # prepare input
            image_1 = Variable(image_1)
            image_2 = Variable(image_2)
            target_res = Variable(label_res)

            if config.use_gpu:
                image_1 = image_1.cuda()
                image_2 = image_2.cuda()
                target_res = target_res.cuda()

            # go through the model
            score_1, score_2, score_res = model(image_1, image_2)

            # backpropagate
            optimizer.zero_grad()
            pair_loss = pair_criterion(score_res, target_res)
            pair_loss.backward()
            optimizer.step()

            pair_loss_meter.add(pair_loss.data[0])
            pair_epoch_loss.add(pair_loss.data[0])

            pair_train_cm.add(softmax(score_res, dim=1).data, target_res.data)

            if (i+1) % config.print_freq == 0:
                vis.plot('loss', pair_loss_meter.value()[0])

        # print result
        pair_train_se = [100. * pair_train_cm.value()[0][0] / (pair_train_cm.value()[0][0] + pair_train_cm.value()[0][1]),
                         100. * pair_train_cm.value()[1][1] / (pair_train_cm.value()[1][0] + pair_train_cm.value()[1][1])]
        model.eval()
        pair_val_cm, pair_val_accuracy, pair_val_se = val_pair(model, valpair_dataloader)

        if np.average(pair_val_se) > pair_previous_avg_se:  # 当测试集上的平均sensitivity升高时保存模型
            if config.parallel:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
                save_model_name = config.save_model_name if config.save_model_name else model.module.model_name + '_best_model.pth'
                if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                    os.makedirs(os.path.join('checkpoints', save_model_dir))
                model.module.save(os.path.join('checkpoints', save_model_dir, save_model_name))
            else:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
                save_model_name = config.save_model_name if config.save_model_name else model.model_name + '_best_model.pth'
                if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                    os.makedirs(os.path.join('checkpoints', save_model_dir))
                model.save(os.path.join('checkpoints', save_model_dir, save_model_name))
            pair_previous_avg_se = np.average(pair_val_se)

        if epoch+1 == config.max_epoch:  # 保存最后一个模型
            if config.parallel:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
                save_model_name = config.save_model_name.split('.pth')[0]+'_last.pth' if config.save_model_name else model.module.model_name + '_last_model.pth'
            else:
                save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
                save_model_name = config.save_model_name.split('.pth')[0]+'_last.pth' if config.save_model_name else model.model_name + '_last_model.pth'
            if not os.path.exists(os.path.join('checkpoints', save_model_dir)):
                os.makedirs(os.path.join('checkpoints', save_model_dir))
            model.save(os.path.join('checkpoints', save_model_dir, save_model_name))

        vis.plot_many({'epoch_loss': pair_epoch_loss.value()[0],
                       'pair_train_avg_se': np.average(pair_train_se), 'pair_train_se_0': pair_train_se[0], 'pair_train_se_1': pair_train_se[1],
                       'pair_val_avg_se': np.average(pair_val_se), 'pair_val_se_0': pair_val_se[0], 'pair_val_se_1': pair_val_se[1]})
        vis.log(f"epoch: [{epoch+1}/{config.max_epoch}] ===============================================")
        vis.log(f"lr: {lr}, loss: {round(pair_epoch_loss.value()[0], 5)}")
        vis.log(f"pair_train_avg_se: {round(np.average(pair_train_se), 4)}, pair_train_se_0: {round(pair_train_se[0], 4)}, pair_train_se_1: {round(pair_train_se[1], 4)}")
        vis.log(f"pair_val_avg_se: {round(sum(pair_val_se) / len(pair_val_se), 4)}, pair_val_se_0: {round(pair_val_se[0], 4)}, pair_val_se_1: {round(pair_val_se[1], 4)}")
        vis.log(f'pair_train_cm: {pair_train_cm.value()}')
        vis.log(f'pair_val_cm: {pair_val_cm.value()}')
        print("lr:", lr, "loss:", round(pair_epoch_loss.value()[0], 5))
        print('pair_train_avg_se:', round(np.average(pair_train_se), 4), 'pair_train_se_0:', round(pair_train_se[0], 4), 'pair_train_se_1:', round(pair_train_se[1], 4))
        print('pair_val_avg_se:', round(np.average(pair_val_se), 4), 'pair_val_se_0:', round(pair_val_se[0], 4), 'pair_val_se_1:', round(pair_val_se[1], 4))
        print('pair_train_cm:')
        print(pair_train_cm.value())
        print('pair_val_cm:')
        print(pair_val_cm.value())

        # update learning rate
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * config.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # previous_loss = loss_meter.value()[0]
        if (epoch+1) % 5 == 0:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def val_cls(model, dataloader):
    val_cm = meter.ConfusionMeter(config.num_classes)
    softmax = functional.softmax

    for i, (image_1, _, label_1, _, _, _, _) in tqdm(enumerate(dataloader)):
        image_1 = Variable(image_1, volatile=True)
        target_1 = Variable(label_1)
        if config.use_gpu:
            image_1 = image_1.cuda()
            target_1 = target_1.cuda()

        score_1, score_2, _ = model(image_1, image_1)
        assert (score_1.cpu().data.numpy() == score_2.cpu().data.numpy()).all()

        val_cm.add(softmax(score_1, dim=1).data, target_1.data)

    val_accuracy = 100. * sum([val_cm.value()[c][c] for c in range(config.num_classes)]) / val_cm.value().sum()
    val_se = [100. * val_cm.value()[0][0] / (val_cm.value()[0][0] + val_cm.value()[0][1]),
              100. * val_cm.value()[1][1] / (val_cm.value()[1][0] + val_cm.value()[1][1])]

    return val_cm, val_accuracy, val_se


def val_pair(model, dataloader):
    val_cm = meter.ConfusionMeter(config.num_classes)
    softmax = functional.softmax

    for i, (image_1, image_2, _, _, label_res, _, _) in tqdm(enumerate(dataloader)):
        image_1 = Variable(image_1, volatile=True)
        image_2 = Variable(image_2, volatile=True)
        target_res = Variable(label_res)
        if config.use_gpu:
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            target_res = target_res.cuda()

        _, _, score_res = model(image_1, image_2)

        val_cm.add(softmax(score_res, dim=1).data, target_res.data)

    val_accuracy = 100. * sum([val_cm.value()[c][c] for c in range(config.num_classes)]) / val_cm.value().sum()
    val_se = [100. * val_cm.value()[0][0] / (val_cm.value()[0][0] + val_cm.value()[0][1]),
              100. * val_cm.value()[1][1] / (val_cm.value()[1][0] + val_cm.value()[1][1])]

    return val_cm, val_accuracy, val_se


def test(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = PairSWDataset(config.test_paths, phase='test', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Test Image:', test_data.__len__())

    # prepare model
    model = SiameseNet(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
        print('Model has been loaded!')
    else:
        print("Don't load model")
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])
    model.eval()

    test_cm = meter.ConfusionMeter(config.num_classes)
    softmax = functional.softmax
    results = []

    # go through the model
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):
        img = Variable(image, volatile=True)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target = target.cuda()

        score = model(img)

        test_cm.add(softmax(score, dim=1).data, target.data)

        for l, p, ip in zip(label, softmax(score, dim=1).data, image_path):
            if p[0] >= p[1]:
                results.append((ip, l, 0, round(p[0], 4), round(p[1], 4)))
            else:
                results.append((ip, l, 1, round(p[0], 4), round(p[1], 4)))

        # for p, ip in zip(softmax(score, dim=1).data, image_path):
        #     # print(p)
        #     b = ip.split('/')[-1].split('.')[0].split('_')[2:6]
        #     if p[1] >= 0.5:
        #         if ip.split('/')[-2] in positive_bbox.keys():
        #             positive_bbox[ip.split('/')[-2]].append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), p[1]))
        #         else:
        #             positive_bbox[ip.split('/')[-2]] = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), p[1])]
        #     else:
        #         pass

    ACC = 100. * sum([test_cm.value()[c][c] for c in range(config.num_classes)]) / test_cm.value().sum()
    SE = 100. * test_cm.value()[1][1] / (test_cm.value()[1][0] + test_cm.value()[1][1])

    print('confusion matrix:')
    print(test_cm.value())
    print('test accuracy:', ACC)
    print('Sensitivity:', SE)

    if config.result_file:
        write_csv(os.path.join('results', config.result_file), tag=['path', 'label', 'predict', 'p1', 'p2'], content=results)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'train_pair': train_pair,
        'test': test,
    })
