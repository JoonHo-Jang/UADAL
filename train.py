from __future__ import print_function
from data_loader.get_loader import get_dataset_information, get_loader
import random
from utils import utils as utils
from models.basenet import *
import datetime
import numpy as np
import time
import datetime
import warnings
from data_loader.base import UDADataset
import os
import sys
warnings.filterwarnings("ignore")


def main(args):
    t1 = time.time()
    sum_str = ''
    args_list = [str(arg) for arg in vars(args)]
    args_list.sort()
    for arg in args_list:
        sum_str += '{:>20} : {:<20} \n'.format(arg, getattr(args, arg))
    utils.setGPU(args.set_gpu)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda:%s" % (0))

    torch.set_num_threads(1)

    source_data, target_data, evaluation_data, num_class = get_dataset_information(args.dataset, args.source_domain, args.target_domain)
    train_transforms, test_transforms = utils.bring_data_transformation()

    src_dset = UDADataset(source_data, source_data, num_class, train_transforms, test_transforms, is_target=False, batch_size=args.batch_size)
    src_train_dset, _, _ = src_dset.get_dsets()
    target_dset = UDADataset(target_data, target_data, num_class, train_transforms, test_transforms, is_target=True, batch_size=args.batch_size)
    target_dset.get_dsets()

    if args.dataset == 'visda':
        dataset_info = '%s' % (args.dataset)
    else:
        dataset_info = '%s_%s_%s' % (args.dataset, args.source_domain, args.target_domain)
    save_info = '%s#%s#%s#%s#%s' % (args.exp_code, args.model, args.net, dataset_info, args.seed)

    result_model_dir = utils.get_save_dir(args, save_info)

    logger = utils.bring_logger(result_model_dir)
    args.logger = logger
    args.logger.info(sum_str)
    args.logger.info('=' * 30)

    if args.model == 'UADAL':
        from models.model_UADAL import UADAL
        model = UADAL(args, num_class, src_dset, target_dset)
    elif args.model == 'cUADAL':
        from models.model_cUADAL import cUADAL
        model = cUADAL(args, num_class, src_dset, target_dset)

    model.train_init()
    model.test(0)
    model.build_model()
    model.train()

if __name__ == '__main__':
    from config import args
    main(args)
