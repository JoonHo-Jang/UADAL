import torch.optim as opt
from models.basenet import *
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter


## Some functions from https://github.com/ksaito-ut/OPDA_BP/blob/master/utils/utils.py

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def StepwiseLRscheduler(step, initial_lr, gamma=10, decay_rate=0.75, max_iter=1000):
    return initial_lr * (1 + gamma * step) ** (-decay_rate)
    # return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def ConstantScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr
def CosineScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    cos = (1 + np.cos((step / max_iter) * np.pi)) / 2
    lr = initial_lr * cos
    return lr
def StepScheduler(step, initial_lr, gamma=500, power=0.2, max_iter=1000):
    divide = step // 500
    lr = initial_lr * (0.2 ** divide)
    return lr
def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    return NGPU

def get_model_init(args, known_num_class, all_num_class):
    net = args.net
    if net == 'vgg':
        model_g = VGGBase()
    elif 'resnet' in net:
        model_g = ResNetFc(model_name=args.net)
    elif 'effi' in net:
        model_g = EfficientNetB0(model_name='efficientnet')
    elif 'densenet' in net:
        model_g = DenseNet(model_name='densenet')
    else:
        print('Please specify the backbone network')
        sys.exit()

    model_e = Net_CLS(in_dim=model_g.output_num(), out_dim=known_num_class, bias=False)
    model_c = Net_CLS_C(in_dim=model_g.output_num(), out_dim=all_num_class, bottle_neck_dim=args.bottle_neck_dim)

    return model_g, model_e, model_c

def get_model(args, known_num_class, all_num_class, domain_dim=3, dc_out_dim=None):
    net = args.net

    if net == 'vgg':
        model_g = VGGBase()
    elif 'resnet' in net:
        model_g = ResNetFc(model_name=args.net)
    elif 'effi' in net:
        model_g = EfficientNetB0(model_name='efficientnet')
    elif 'densenet' in net:
        model_g = DenseNet(model_name='densenet')
    else:
        print('Please specify the backbone network')
        sys.exit()

    if dc_out_dim is None:
        dc_dim = model_g.output_num()
    else:
        dc_dim = dc_out_dim

    model_e = Net_CLS(in_dim=model_g.output_num(), out_dim=known_num_class, bias=False)
    model_c = Net_CLS_C(in_dim=model_g.output_num(), out_dim=all_num_class, bottle_neck_dim=args.bottle_neck_dim)
    model_dc = Net_CLS_DC(dc_dim, out_dim=domain_dim, bottle_neck_dim=args.bottle_neck_dim2)

    return model_g, model_c, model_e, model_dc

class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1

def bring_data_transformation():

    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])

    return train_transforms, test_transforms


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def get_save_dir(args, save_info):
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_folder_dir = os.path.join(result_dir, args.exp_code)
    if not os.path.exists(result_folder_dir):
        os.mkdir(result_folder_dir)
    result_saveinfo_dir = os.path.join(result_folder_dir, '%s.txt'%save_info)
    return result_saveinfo_dir


def bring_logger(results_log_dir, level='info'):
    import logging
    log1 = logging.getLogger('model specific logger')
    streamH = logging.StreamHandler()
    log1.addHandler(streamH)
    fileH = logging.FileHandler(results_log_dir)
    log1.addHandler(fileH)
    if level =='debug':
        log1.setLevel(level=logging.DEBUG)
    else:
        log1.setLevel(level=logging.INFO)
    return log1

from PIL import Image
def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

