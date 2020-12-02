# coding: utf-8
import os
import cv2
import fire
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from torch.nn import functional
from torchnet import meter

from config import config
from dataset import SlideWindowDataset
from models import UNet_Classifier
from utils import Visualizer, write_csv, write_json, dice_coeff


def train(**kwargs):
    config.parse(kwargs)

    # ============================================ Visualization =============================================
    vis = Visualizer(port=2333, env=config.env)
    vis.log('Use config:')
    for k, v in config.__class__.__dict__.items():
        if not k.startswith('__'):
            vis.log(f"{k}: {getattr(config, k)}")

    # ============================================= Prepare Data =============================================
    train_data = SlideWindowDataset(config.train_paths, phase='train', useRGB=config.useRGB, usetrans=config.usetrans, balance=config.data_balance)
    val_data = SlideWindowDataset(config.test_paths, phase='val', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    print('Training Images:', train_data.__len__(), 'Validation Images:', val_data.__len__())
    dist = train_data.dist()
    print('Train Data Distribution:', dist)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # ============================================= Prepare Model ============================================
    model = UNet_Classifier(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
        print('Model loaded')
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])

    # =========================================== Criterion and Optimizer =====================================
    # weight = torch.Tensor([1, 1])
    # weight = torch.Tensor([dist['1']/(dist['0']+dist['1']), dist['0']/(dist['0']+dist['1'])])  # weight需要将二者反过来，多于二分类可以取倒数
    # weight = torch.Tensor([1, 3.5])
    # weight = torch.Tensor([1, 5])
    weight = torch.Tensor([1, 7])

    vis.log(f'loss weight: {weight}')
    print('loss weight:', weight)
    weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # ================================================== Metrics ===============================================
    softmax = functional.softmax
    loss_meter_edge = meter.AverageValueMeter()
    epoch_loss_edge = meter.AverageValueMeter()
    loss_meter_cls = meter.AverageValueMeter()
    epoch_loss_cls = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    epoch_loss = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(config.num_classes)

    # ====================================== Saving and Recording Configuration =================================
    previous_auc = 0
    if config.parallel:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.module.model_name + '_best_model.pth'
    else:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.model_name + '_best_model.pth'
    save_epoch = 1  # 用于记录验证集上效果最好模型对应的epoch
    process_record = {'epoch_loss': [], 'epoch_loss_edge': [], 'epoch_loss_cls': [],
                      'train_avg_se': [], 'train_se_0': [], 'train_se_1': [],
                      'val_avg_se': [], 'val_se_0': [], 'val_se_1': [],
                      'AUC': [], 'DICE': []}  # 用于记录实验过程中的曲线，便于画曲线图

    # ================================================== Training ===============================================
    for epoch in range(config.max_epoch):
        print(f"epoch: [{epoch + 1}/{config.max_epoch}] {config.save_model_name[:-4]} ==================================")
        train_cm.reset()
        epoch_loss.reset()
        dice = []

        # ****************************************** train ****************************************
        model.train()
        for i, (image, label, edge_mask, image_path) in tqdm(enumerate(train_dataloader)):
            loss_meter.reset()

            # ------------------------------------ prepare input ------------------------------------
            if config.use_gpu:
                image = image.cuda()
                label = label.cuda()
                edge_mask = edge_mask.cuda()

            # ---------------------------------- go through the model --------------------------------
            score, score_mask = model(x=image)

            # ----------------------------------- backpropagate -------------------------------------
            optimizer.zero_grad()

            # 分类loss
            loss_cls = criterion(score, label)
            # 对Edge包含pixel加loss
            log_prob_mask = functional.logsigmoid(score_mask)
            count_edge = torch.sum(edge_mask, dim=(1, 2, 3), keepdim=True)
            loss_edge = -1 * torch.mean(torch.sum(edge_mask * log_prob_mask, dim=(1, 2, 3), keepdim=True) / (count_edge + 1e-8))

            # 对非Edge包含pixel加loss
            r_prob_mask = torch.Tensor([1.0]).cuda() - torch.sigmoid(score_mask)
            r_edge_mask = torch.Tensor([1.0]).cuda() - edge_mask
            log_rprob_mask = torch.log(r_prob_mask + 1e-5)
            count_redge = torch.sum(r_edge_mask, dim=(1, 2, 3), keepdim=True)
            loss_redge = -1 * torch.mean(torch.sum(r_edge_mask * log_rprob_mask, dim=(1, 2, 3), keepdim=True) / (count_redge + 1e-8))

            # 权重按照前景和背景的像素点数量来算
            w1 = torch.sum(count_edge).item() / (torch.sum(count_edge).item() + torch.sum(count_redge).item())
            w2 = torch.sum(count_redge).item() / (torch.sum(count_edge).item() + torch.sum(count_redge).item())
            loss = loss_cls + w1 * loss_edge + w2 * loss_redge

            loss.backward()
            optimizer.step()

            # ------------------------------------ record loss ------------------------------------
            loss_meter_edge.add((w1 * loss_edge + w2 * loss_redge).item())
            epoch_loss_edge.add((w1 * loss_edge + w2 * loss_redge).item())
            loss_meter_cls.add(loss_cls.item())
            epoch_loss_cls.add(loss_cls.item())
            loss_meter.add(loss.item())
            epoch_loss.add(loss.item())
            train_cm.add(softmax(score, dim=1).detach(), label.detach())
            dice.append(dice_coeff(input=(score_mask > 0.5).float(), target=edge_mask[:, 0, :, :]).item())

            if (i + 1) % config.print_freq == 0:
                vis.plot_many({'loss': loss_meter.value()[0],
                               'loss_edge': loss_meter_edge.value()[0],
                               'loss_cls': loss_meter_cls.value()[0]})

        train_se = [100. * train_cm.value()[0][0] / (train_cm.value()[0][0] + train_cm.value()[0][1]),
                    100. * train_cm.value()[1][1] / (train_cm.value()[1][0] + train_cm.value()[1][1])]
        train_dice = sum(dice) / len(dice)

        # *************************************** validate ***************************************
        model.eval()
        if (epoch + 1) % 1 == 0:
            Best_T, val_cm, val_spse, val_accuracy, AUC, val_dice = val(model, val_dataloader)

            # ------------------------------------ save model ------------------------------------
            if AUC > previous_auc and epoch + 1 > 5:  # 5个epoch之后，当测试集上的平均sensitivity升高时保存模型
                if config.parallel:
                    if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                        os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                    model.module.save(
                        os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                else:
                    if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                        os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                    model.save(os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                previous_auc = AUC
                save_epoch = epoch + 1

            # ---------------------------------- recond and print ---------------------------------
            process_record['epoch_loss'].append(epoch_loss.value()[0])
            process_record['epoch_loss_edge'].append(epoch_loss_edge.value()[0])
            process_record['epoch_loss_cls'].append(epoch_loss_cls.value()[0])
            process_record['train_avg_se'].append(np.average(train_se))
            process_record['train_se_0'].append(train_se[0])
            process_record['train_se_1'].append(train_se[1])
            process_record['val_avg_se'].append(np.average(val_spse))
            process_record['val_se_0'].append(val_spse[0])
            process_record['val_se_1'].append(val_spse[1])
            process_record['AUC'].append(AUC)
            process_record['DICE'].append(val_dice)

            vis.plot_many({'epoch_loss': epoch_loss.value()[0], 'epoch_loss_edge': epoch_loss_edge.value()[0],
                           'epoch_loss_cls': epoch_loss_cls.value()[0],
                           'train_avg_se': np.average(train_se), 'train_se_0': train_se[0], 'train_se_1': train_se[1],
                           'val_avg_se': np.average(val_spse), 'val_se_0': val_spse[0], 'val_se_1': val_spse[1],
                           'AUC': AUC, 'train_dice': train_dice, 'val_dice': val_dice})
            vis.log(f"epoch: [{epoch + 1}/{config.max_epoch}] ===============================================")
            vis.log(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(loss_meter.value()[0], 5)}")
            vis.log(f"train_avg_se: {round(np.average(train_se), 4)}, train_se_0: {round(train_se[0], 4)}, train_se_1: {round(train_se[1], 4)}")
            vis.log(f"train_dice: {round(train_dice, 4)}")
            vis.log(f"val_avg_se: {round(sum(val_spse) / len(val_spse), 4)}, val_se_0: {round(val_spse[0], 4)}, val_se_1: {round(val_spse[1], 4)}")
            vis.log(f"val_dice: {round(val_dice, 4)}")
            vis.log(f"AUC: {AUC}")
            vis.log(f'train_cm: {train_cm.value()}')
            vis.log(f'Best Threshold: {Best_T}')
            vis.log(f'val_cm: {val_cm}')
            print("lr:", optimizer.param_groups[0]['lr'], "loss:", round(epoch_loss.value()[0], 5))
            print('train_avg_se:', round(np.average(train_se), 4), 'train_se_0:', round(train_se[0], 4), 'train_se_1:', round(train_se[1], 4))
            print('train_dice:', train_dice)
            print('val_avg_se:', round(np.average(val_spse), 4), 'val_se_0:', round(val_spse[0], 4), 'val_se_1:', round(val_spse[1], 4))
            print('val_dice:', val_dice)
            print('AUC:', AUC)
            print('train_cm:')
            print(train_cm.value())
            print('Best Threshold:', Best_T, 'val_cm:')
            print(val_cm)

            # ------------------------------------ save record ------------------------------------
            if os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0])):
                write_json(file=os.path.join('checkpoints', save_model_dir, save_model_name[:-4], 'process_record.json'), content=process_record)
        # if (epoch+1) % 20 == 0:
        #     lr = lr * config.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

    vis.log(f"Best Epoch: {save_epoch}")
    print("Best Epoch:", save_epoch)


def val(model, dataloader):
    # ============================ Prepare Metrics ==========================
    T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cm = {'0.1': [[0, 0], [0, 0]],
          '0.2': [[0, 0], [0, 0]],
          '0.3': [[0, 0], [0, 0]],
          '0.4': [[0, 0], [0, 0]],
          '0.5': [[0, 0], [0, 0]],
          '0.6': [[0, 0], [0, 0]],
          '0.7': [[0, 0], [0, 0]],
          '0.8': [[0, 0], [0, 0]],
          '0.9': [[0, 0], [0, 0]]}
    dice = []
    softmax = functional.softmax

    # ================================ Validate ==============================
    for i, (image, label, edge_mask, image_path) in tqdm(enumerate(dataloader)):
        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()
            edge_mask = edge_mask.cuda()

        score, score_mask = model(x=image)

        # *********************** confusion matrix and dice ***********************
        for p, l in zip(softmax(score, dim=1).detach(), label.detach()):
            for t in T:
                if p[1] >= t:
                    cm[str(t)][int(l)][1] += 1
                else:
                    cm[str(t)][int(l)][0] += 1
        dice.append(dice_coeff(input=(score_mask > 0.5).float(), target=edge_mask[:, 0, :, :]).item())

    # ============================ Calculate ROC Curve and Best Threshold ==========================
    ROC = {str(t): [cm[str(t)][0][0] / (cm[str(t)][0][0] + cm[str(t)][0][1]),
                    cm[str(t)][1][1] / (cm[str(t)][1][0] + cm[str(t)][1][1])] for t in T}
    Best_T = sorted(ROC.items(), key=lambda x: x[1][0] + x[1][1] - 1, reverse=True)[0][0]
    val_accuracy = 100. * sum([cm[Best_T][c][c] for c in range(config.num_classes)]) / np.sum(cm[Best_T])
    val_spse = [100. * cm[Best_T][0][0] / (cm[Best_T][0][0] + cm[Best_T][0][1]),
                100. * cm[Best_T][1][1] / (cm[Best_T][1][0] + cm[Best_T][1][1])]

    # ==================================== Calculate AUC ===========================================
    AUC = 0
    for i in range(len(T) - 1):
        AUC += (ROC[str(T[i])][1] + ROC[str(T[i + 1])][1]) / 2 * (ROC[str(T[i + 1])][0] - ROC[str(T[i])][0])
    AUC += (1 + ROC[str(T[0])][1]) * ROC[str(T[0])][0] / 2
    AUC += ROC[str(T[-1])][1] * (1 - ROC[str(T[-1])][0]) / 2

    val_dice = sum(dice) / len(dice)

    return Best_T, cm[Best_T], val_spse, val_accuracy, AUC, val_dice


def test(**kwargs):
    config.parse(kwargs)

    # ============================================= Prepare Data =============================================
    test_data = SlideWindowDataset(config.test_paths, phase='test', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    # test_data = SlideWindowDataset(config.train_paths, phase='test_train', useRGB=config.useRGB, usetrans=config.usetrans, balance=False)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print('Test Image:', test_data.__len__())

    # ============================================= Prepare Model ============================================
    model = UNet_Classifier(num_classes=config.num_classes)
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

    # =========================================== Prepare Metrics =====================================
    T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cm = {'0.1': [[0, 0], [0, 0]],
          '0.2': [[0, 0], [0, 0]],
          '0.3': [[0, 0], [0, 0]],
          '0.4': [[0, 0], [0, 0]],
          '0.5': [[0, 0], [0, 0]],
          '0.6': [[0, 0], [0, 0]],
          '0.7': [[0, 0], [0, 0]],
          '0.8': [[0, 0], [0, 0]],
          '0.9': [[0, 0], [0, 0]]}
    softmax = functional.softmax
    results = []

    # =========================================== Test ============================================
    # for i, (image, label, edge_mask, image_path) in tqdm(enumerate(test_dataloader)):
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):
        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()
            # edge_mask = edge_mask.cuda()

        score, score_mask, cam_0, cam_1 = model(x=image)
        # score = model(image)
        # print(cam_0.size())
        prob = softmax(score, dim=1).detach()

        # ******************************** confusion matrix ***************************
        for p, l in zip(prob, label.detach()):
            for t in T:
                if p[1] >= t:
                    cm[str(t)][int(l)][1] += 1
                else:
                    cm[str(t)][int(l)][0] += 1

        # ******************************** generate CAM ******************************
        # 按照预测将结果选择CAM0和CAM1其中的一个计算并保存
        # for n in range(cam_0.size(0)):
        #     if prob[n][1] < 0.2:
        #         cam = cam_0.cpu().detach().numpy()[n][0]
        #     else:
        #         cam = cam_1.cpu().detach().numpy()[n][0]
        #     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        #     img = cv2.resize(cv2.imread(image_path[n], 1), (224, 224))
        #     # img = cv2.resize(cv2.imread(image_path[n], 1), (448, 448))
        #     cam = np.float32(heatmap) / 255 + np.float32(img) / 255
        #     cam = np.uint8(255 * cam / np.max(cam))
        #
        #     img_split_path = image_path[n].split('/')
        #     cam_save_path = '/'.join([*img_split_path[:8], 'CAM', 'CAM_' + config.load_model_path.split('/')[-1][:-4] + '-' + img_split_path[8], *img_split_path[9:11]])
        #     if not os.path.exists(cam_save_path):
        #         os.makedirs(cam_save_path)
        #     cam_save_name = img_split_path[-1][:-4] + '_' + str(round(prob.cpu().numpy()[n][int(label[n])], 4)) + '.png'
        #
        #     cv2.imwrite(os.path.join(cam_save_path, cam_save_name), cam)

        # 同时计算CAM0和CAM1并保存
        for n in range(cam_0.size(0)):
            cam = cam_0.cpu().detach().numpy()[n][0]
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img = cv2.resize(cv2.imread(image_path[n], 1), (224, 224))
            # img = cv2.resize(cv2.imread(image_path[n], 1), (448, 448))
            cam = np.float32(heatmap) / 255 + np.float32(img) / 255
            cam = np.uint8(255 * cam / np.max(cam))

            img_split_path = image_path[n].split('/')
            cam_save_path = '/'.join([*img_split_path[:8], 'CAM',
                                      'CAM0_' + config.load_model_path.split('/')[-1][:-4] + '-' + img_split_path[8],
                                      *img_split_path[9:11]])
            if not os.path.exists(cam_save_path):
                os.makedirs(cam_save_path)
            cam_save_name = img_split_path[-1][:-4] + '_' + str(round(prob.cpu().numpy()[n][0], 4)) + '.png'
            cv2.imwrite(os.path.join(cam_save_path, cam_save_name), cam)

        for n in range(cam_1.size(0)):
            cam = cam_1.cpu().detach().numpy()[n][0]
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img = cv2.resize(cv2.imread(image_path[n], 1), (224, 224))
            # img = cv2.resize(cv2.imread(image_path[n], 1), (448, 448))
            cam = np.float32(heatmap) / 255 + np.float32(img) / 255
            cam = np.uint8(255 * cam / np.max(cam))

            img_split_path = image_path[n].split('/')
            cam_save_path = '/'.join([*img_split_path[:8], 'CAM',
                                      'CAM1_' + config.load_model_path.split('/')[-1][:-4] + '-' + img_split_path[8],
                                      *img_split_path[9:11]])
            if not os.path.exists(cam_save_path):
                os.makedirs(cam_save_path)
            cam_save_name = img_split_path[-1][:-4] + '_' + str(round(prob.cpu().numpy()[n][1], 4)) + '.png'
            cv2.imwrite(os.path.join(cam_save_path, cam_save_name), cam)

        # ******************************** record prediction results ******************************
        for l, p, ip in zip(label.detach(), prob, image_path):
            if p[1] < 0.1:
                results.append((ip, int(l), 0, round(float(p[0]), 4), round(float(p[1]), 4)))
            else:
                results.append((ip, int(l), 1, round(float(p[0]), 4), round(float(p[1]), 4)))

    # ============================ Calculate ROC Curve and Best Threshold ==========================
    ROC = {str(t): [cm[str(t)][0][0] / (cm[str(t)][0][0] + cm[str(t)][0][1]),
                    cm[str(t)][1][1] / (cm[str(t)][1][0] + cm[str(t)][1][1])] for t in T}
    Best_T = sorted(ROC.items(), key=lambda x: x[1][0] + x[1][1] - 1, reverse=True)[0][0]
    test_accuracy = 100. * sum([cm[Best_T][c][c] for c in range(config.num_classes)]) / np.sum(cm[Best_T])
    test_spse = [100. * cm[Best_T][0][0] / (cm[Best_T][0][0] + cm[Best_T][0][1]),
                 100. * cm[Best_T][1][1] / (cm[Best_T][1][0] + cm[Best_T][1][1])]

    # ======================================== Calculate AUC ======================================
    AUC = 0
    for i in range(len(T) - 1):
        AUC += (ROC[str(T[i])][1] + ROC[str(T[i + 1])][1]) / 2 * (ROC[str(T[i + 1])][0] - ROC[str(T[i])][0])
    AUC += (1 + ROC[str(T[0])][1]) * ROC[str(T[0])][0] / 2
    AUC += ROC[str(T[-1])][1] * (1 - ROC[str(T[-1])][0]) / 2

    # ============================== Print Results and Draw ROC Curve ==============================
    pprint(ROC)
    print('Best T:', Best_T)
    print('confusion matrix:')
    print(cm[Best_T])
    print('test accuracy:', test_accuracy)
    print('average sensitivity:', np.average(test_spse))
    print('sensitivity:', test_spse[0], test_spse[1])
    print('AUC:', AUC)

    # 画ROC曲线
    # x1, y1 = [], []
    # for item in sorted(ROC.items(), key=lambda x: float(x[0]), reverse=True):
    #     x1.append(1 - float(item[1][0]))
    #     y1.append(float(item[1][1]))
    # x1.insert(0, 0)
    # y1.insert(0, 0)
    # x1.append(1)
    # y1.append(1)
    #
    # x2 = [0, 2 - ROC[Best_T][0] - ROC[Best_T][1]]
    # y2 = [ROC[Best_T][0] + ROC[Best_T][1] - 1, 1]
    #
    # plt.title('ROC')
    # plt.xlabel('1 - SP')
    # plt.ylabel('SE')
    #
    # plt.plot(x1, y1, color='red', marker='o', mec='r', mfc='r', label='ROC')
    # plt.plot(x2, y2, color='blue', marker='*', mec='b', mfc='b')
    # plt.grid()
    # plt.savefig(os.path.join('results', config.load_model_path.split('/')[-1][:-4] + "_ROC.png"))

    # ===================================== Save Prediction Results ===========================
    if config.result_file:
        write_csv(os.path.join('results', config.result_file), tag=['path', 'label', 'predict', 'p1', 'p2'],
                  content=results)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })
