from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys
import numpy as np
from collections import Counter
#torch.set_num_threads()
import warnings
warnings.filterwarnings("ignore")

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/data_loader/get_loader.py

def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32):
    sampler = None
    pin = True
    num_workers = 2

    source_folder_train = ImageFolder(os.path.join(source_path), transform=transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path), transform=transforms[source_path])
    target_folder_test = ImageFolder(os.path.join(evaluation_path), transform=transforms[evaluation_path])
    source_folder_test = ImageFolder(os.path.join(source_path), transform=transforms[source_path])

    freq = Counter(source_folder_train.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder_train.labels]
    sampler = WeightedRandomSampler(source_weights,
                                    len(source_folder_train.labels))
    aligned_train_loader = UnalignedDataLoader()
    aligned_train_loader.initialize(source_folder_train, target_folder_train, batch_size, sampler=sampler)

    target_train_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)
    target_test_loader = torch.utils.data.DataLoader(
        target_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)
    source_test_loader = torch.utils.data.DataLoader(
        source_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)

    if sampler is not None:
        source_train_loader = torch.utils.data.DataLoader(source_folder_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin, drop_last=True)
    else:
        source_train_loader = torch.utils.data.DataLoader(source_folder_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)

    return aligned_train_loader, target_train_loader, target_test_loader, source_train_loader, source_test_loader

def get_dataset_information(dataset, s_d, t_d):
    if dataset == 'office':
        name_dict = {'A':'amazon', 'D':'dslr', 'W':'webcam'}
        data_path = os.path.join('data', 'office')
        source_path = os.path.join(data_path, 'office_%s_source_list.txt'%name_dict[s_d])
        target_path = os.path.join(data_path, 'office_%s_target_list.txt'%name_dict[t_d])
        evaluation_data = os.path.join(data_path, 'office_%s_target_list.txt'%name_dict[t_d])
        class_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', "unk"]
        num_class = len(class_list) #11
    elif dataset == 'officehome':
        name_dict = {'A': 'Art', 'C': 'Clipart', 'P': 'Product', 'R':'Real World'}
        data_path = os.path.join('data', 'officehome')
        source_path = os.path.join(data_path, 'officehome_%s_source_list.txt'%name_dict[s_d])
        target_path = os.path.join(data_path, 'officehome_%s_target_list.txt'%name_dict[t_d])
        evaluation_data = os.path.join(data_path, 'officehome_%s_target_list.txt'%name_dict[t_d])
        class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                      'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                      'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                      'unk']
        num_class = len(class_list) #26
    elif dataset =='visda':
        s_d, t_d = 'train', 'validation'
        data_path = os.path.join('data', dataset)
        source_path = os.path.join(data_path, 'source_list.txt')
        target_path = os.path.join(data_path, 'target_list.txt')
        evaluation_data = os.path.join(data_path, 'target_list.txt')
        class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
        num_class = len(class_list) #7
    else:
        print('Specify the name of dataset!!')
        sys.exit()
    print(source_path)
    print(target_path)
    return source_path, target_path, evaluation_data, num_class