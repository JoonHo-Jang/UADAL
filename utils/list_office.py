import os
import random

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/utils/list_visda.py

class_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet']#, "unk"]
unknown_source_list = ['headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen']#not_shared_list[:10]
unknown_target_list = ['phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']#not_shared_list[10:]

class_list_10 = class_list
office11_target_class_list = class_list + unknown_target_list
office21_target_class_list = class_list + unknown_source_list + unknown_target_list

data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

data_path = os.path.join('../datasets', 'office')
save_path = os.path.join(data_dir, 'office')
if not os.path.exists(save_path):
    os.mkdir(save_path)

for domain in ['amazon', 'dslr', 'webcam']:
    data_domain_path = os.path.join(data_path, domain)
    image_path = os.path.join(data_domain_path, 'images')
    dir_list = os.listdir(image_path)

    path_source_domain = os.path.join(save_path, 'office_%s_source_list.txt'%(domain))
    write_source_domain = open(path_source_domain, "w")
    path_target_domain = os.path.join(save_path, 'office_%s_target_list.txt'%(domain))
    write_target_domain = open(path_target_domain, "w")

    for k, direc in enumerate(dir_list):
        if not '.txt' in direc:
            files = os.listdir(os.path.join(image_path, direc))
            for i, file in enumerate(files):
                file_name = os.path.join(image_path, direc, file)
                if direc in class_list_10:
                    class_name = direc
                    write_source_domain.write('%s\t%s\n' % (file_name, class_list_10.index(class_name)))
                    write_target_domain.write('%s\t%s\n' % (file_name, class_list_10.index(class_name)))
                else:
                    if direc in unknown_target_list:
                        class_name = direc
                        write_target_domain.write('%s\t%s\n' % (file_name, office11_target_class_list.index(class_name)))

