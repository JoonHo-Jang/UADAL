import os
import random

import os

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/utils/list_visda.py


data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# data_path = os.path.join('/home/aailab/adkto809/datasets', 'visda')
data_path = os.path.join('../datasets', 'visda')
save_path = os.path.join(data_dir, 'visda')
if not os.path.exists(save_path):
    os.mkdir(save_path)
p_path = os.path.join(data_path, 'train')
dir_list = os.listdir(p_path)
path_source = os.path.join(save_path, 'source_list.txt')
write_source = open(path_source,"w")
print(dir_list)

class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]

visda_target_class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck"] + ['aeroplane', 'horse', 'knife', 'person', 'plant', 'skateboard']
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_source.write('%s %s\n' % (file_name, class_list.index(class_name)))
            else:
                continue

p_path = os.path.join(data_path, 'validation')
dir_list = os.listdir(p_path)
path_target = os.path.join(save_path, 'target_list.txt')
write_target = open(path_target, "w")

print(dir_list)
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in visda_target_class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s\t%s\n' % (file_name, visda_target_class_list.index(class_name)))





