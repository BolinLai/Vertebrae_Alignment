# coding:utf-8
import os
import warnings


class Config(object):
    env = 'Vertebrae_Alignment'

    root = '/DB/rhome/bllai/PyTorchProjects/Vertebrae_Alignment_Torch1.0'

    # train_paths = [os.path.join(root, 'dataset/train_SW_correct.csv')]
    # train_paths = [os.path.join(root, 'dataset/train_blchosen_1.csv')]
    # val_paths = [os.path.join(root, 'dataset/val_SW_correct.csv'),
    #              os.path.join(root, 'dataset/test_SW_correct.csv')]
    # test_paths = [os.path.join(root, 'dataset/val_SW_correct.csv'),
    #               os.path.join(root, 'dataset/test_SW_correct.csv')]

    # train_paths = [os.path.join(root, 'dataset/train_MaskedImage_correct.csv')]
    # train_paths = [os.path.join(root, 'dataset/train_chosen_4.csv')]
    # val_paths = [os.path.join(root, 'dataset/val_MaskedImage_correct.csv'),
    #              os.path.join(root, 'dataset/test_MaskedImage_correct.csv')]
    # test_paths = [os.path.join(root, 'dataset/val_MaskedImage_correct.csv'),
    #               os.path.join(root, 'dataset/test_MaskedImage_correct.csv')]

    # train_paths = [os.path.join(root, 'dataset/train_MaskedImage_revise.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_MaskedImage_revise.csv')]

    # ALL
    # train_paths = [os.path.join(root, 'dataset/train_SW_revise.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_SW_revise.csv')]

    # All Data 5Fold-1
    # train_paths = [os.path.join(root, 'dataset/train_D1F1.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D1F1.csv')]

    # All Data 5Fold-2
    # train_paths = [os.path.join(root, 'dataset/train_D1F2.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D1F2.csv')]

    # All Data 5Fold-3
    # train_paths = [os.path.join(root, 'dataset/train_D1F3.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D1F3.csv')]

    # All Data 5Fold-4
    # train_paths = [os.path.join(root, 'dataset/train_D1F4.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D1F4.csv')]

    # All Data 5Fold-5
    # train_paths = [os.path.join(root, 'dataset/train_D1F5.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D1F5.csv')]

    # 1/2 Data 5Fold-1
    # train_paths = [os.path.join(root, 'dataset/train_D2F1.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D2F1.csv')]

    # 1/2 Data 5Fold-2
    # train_paths = [os.path.join(root, 'dataset/train_D2F2.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D2F2.csv')]

    # 1/2 Data 5Fold-3
    # train_paths = [os.path.join(root, 'dataset/train_D2F3.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D2F3.csv')]

    # 1/2 Data 5Fold-4
    # train_paths = [os.path.join(root, 'dataset/train_D2F4.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D2F4.csv')]

    # 1/2 Data 5Fold-5
    # train_paths = [os.path.join(root, 'dataset/train_D2F5.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D2F5.csv')]

    # 1/4 Data 5Fold-1
    # train_paths = [os.path.join(root, 'dataset/train_D4F1.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D4F1.csv')]

    # 1/4 Data 5Fold-2
    # train_paths = [os.path.join(root, 'dataset/train_D4F2.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D4F2.csv')]

    # 1/4 Data 5Fold-3
    # train_paths = [os.path.join(root, 'dataset/train_D4F3.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D4F3.csv')]

    # 1/4 Data 5Fold-4
    # train_paths = [os.path.join(root, 'dataset/train_D4F4.csv')]
    # test_paths = [os.path.join(root, 'dataset/test_D4F4.csv')]

    # 1/4 Data 5Fold-5
    train_paths = [os.path.join(root, 'dataset/train_D4F5.csv')]
    test_paths = [os.path.join(root, 'dataset/test_D4F5.csv')]

    save_model_dir = None
    save_model_name = None
    load_model_path = None
    result_file = None

    data_balance = False
    useRGB = True
    usetrans = True

    batch_size = 16
    num_workers = 8
    print_freq = 50

    num_classes = 2
    max_epoch = 100
    lr = 0.0001
    lr_decay = 0.95
    weight_decay = 1e-5

    use_gpu = True
    parallel = False
    num_of_gpu = 2

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f'Warning: config has no attribute {k}')
            setattr(self, k, v)

        print('Use config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


config = Config()

# CUDA_VISIBLE_DEVICES=1 python main.py train --env='' --save_model_name=''


# 加入补充数据后
#     Train  Val  Test
#
# P     84    9    20
#
# N     35    9    20

# S4Shift_2010
#     Train     Val    Test
#
# P    6608     689    1952
#
# N   20036    2991    6724

# MaskedImage_2010_corrected
#     Train     Val    Test
#
# P    6721     827    1697
#
# N   18352    3534    7115

# 数据修正后
#     Train  Test
#
# P    78     26
#
# N    38     26

# SW_revise
#     Train     Test
#
# P    6319     2154
#
# N   17753     9710
