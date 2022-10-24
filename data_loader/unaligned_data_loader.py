
import torch.utils.data
from builtins import object

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/data_loader/unaligned_data_loader.py

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        img_A, label_A, ind_A, path_A = None, None, None, None
        img_B, label_B, ind_B, path_B = None, None, None, None
        try:
            img_A, label_A, ind_A, path_A = next(self.data_loader_A_iter)
        except StopIteration:
            if img_A is None or label_A is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                img_A, label_A, ind_A, path_A = next(self.data_loader_A_iter)
        try:
            img_B, label_B, ind_B, path_B = next(self.data_loader_B_iter)
        except StopIteration:
            if img_B is None or label_B is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                img_B, label_B, ind_B, path_B = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'img_s': img_A, 'label_s': label_A, 'ind_s':ind_A, 'path_s':path_A,
                   'img_t': img_B, 'label_t': label_B, 'ind_t':ind_B, 'path_t':path_B}


class UnalignedDataLoader():
    def initialize(self, A, B, batchSize, sampler=None):
        dataset_A = A
        dataset_B = B

        num_workers = 2
        if sampler is not None:
            data_loader_A = torch.utils.data.DataLoader(
                dataset_A,
                batch_size=batchSize,
                sampler=sampler,
                num_workers=num_workers, pin_memory=True)
        else:
            data_loader_A = torch.utils.data.DataLoader(
                dataset_A,
                batch_size=batchSize,
                shuffle=True,
                num_workers=num_workers, pin_memory=True)

        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=batchSize,
            shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False
        self.paired_data = PairedData(data_loader_A, data_loader_B, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.opt.max_dataset_size)
