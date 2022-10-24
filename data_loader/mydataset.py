import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/data_loader/mydataset.py

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset_nolist(image_list):
    with open(image_list) as f:
        image_index = [x.split('\t')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            #print(x)
            label = x.split('\t')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
        # print(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list, selected_list


class ImageFolder(data.Dataset):
    def __init__(self, image_list, transform=None, target_transform=None,
                 loader=default_loader,train=False):
        imgs, labels, idx_list = make_dataset_nolist(image_list)
        #self.root = root
        self.imgs = imgs
        self.labels= labels
        self.idx_list = idx_list
        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        ind = self.idx_list[index]
        #if self.train:
        #    img = augment_images(img)
        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, ind, path

    def __len__(self):
        return len(self.imgs)


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, flist_reader=default_flist_reader,
                 loader=default_loader, return_paths=True):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        impath = impath.replace('other','unk')
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, impath
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


def make_dataset_(image_list, labels=None):
	if labels:
		len_ = len(image_list)
		images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
	else:
		if len(image_list[0].split()) > 2:
			images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
		else:
			images = [(val.split()[0], int(val.split()[1])) for val in image_list]
	return images

class ImageList(object):
    """
    A generic data loader where the images are arranged in this way: ::
            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/xxz.png
            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/asd932_.png
        Args:
            root (string): Root directory path.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable, optional): A function to load an image given its path.
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, image_list, transform=None, target_transform=None,
				 loader=default_loader):
        #imgs = make_dataset_(image_list)
        imgs, labels, idx_list = make_dataset_nolist(image_list)

        self.data = imgs#np.array([os.path.join(img[0]) for img in imgs])
        self.labels = labels#np.array([img[1] for img in imgs])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.data[index], self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)