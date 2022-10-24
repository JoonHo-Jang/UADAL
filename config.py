
from __future__ import print_function
import argparse


parser = argparse.ArgumentParser(description='PyTorch code for UADAL/cUADAL')

# Data Level
parser.add_argument('--dataset', type=str, default='office',
                    help='visda, office, officehome')
parser.add_argument('--source_domain', type=str, default='A',
                    help='A, D, W ')
parser.add_argument('--target_domain', type=str, default='W',
                    help='A, D, W ')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

## Model Level
parser.add_argument('--model', type=str, default='UADAL',
                    help='UADAL, cUADAL')
parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                    help='resnet50, efficientnet, densenet, vgg')
parser.add_argument('--bottle_neck_dim', type=int, default=256, metavar='B',
                    help='bottle_neck_dim for the classifier network.')
parser.add_argument('--bottle_neck_dim2', type=int, default=500, metavar='B',
                    help='bottle_neck_dim for the classifier network.')

## Iteration Level
parser.add_argument('--warmup_iter', type=int, default=2000, metavar='S',
                    help='warmup iteration for posterior inference')
parser.add_argument('--training_iter', type=int, default=100, metavar='S',
                    help='training_iter')
parser.add_argument('--update_term', type=int, default=10, metavar='S',
                    help='update term for posterior inference')

## Loss Level
parser.add_argument('--threshold', type=float, default=0.85, metavar='fixmatch',
                    help='threshold for fixmatch')
parser.add_argument('--ls_eps', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

## Optimization Level
parser.add_argument('--update_freq_D', type=int, default=1, metavar='S',
                    help='freq for D in optimization.')
parser.add_argument('--update_freq_G', type=int, default=1, metavar='S',
                    help='freq for G in optimization.')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--scheduler', type=str, default='cos',
                    help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--e_lr', type=float, default=0.002, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--g_lr', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

parser.add_argument('--opt_clip', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')
## etc:
parser.add_argument('--exp_code', type=str, default='Test', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--result_dir', type=str, default='results', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--set_gpu', type=int, default=0,
                    help='gpu setting 0 or 1')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')

try:
    args = parser.parse_args()
except:
    args, _ = parser.parse_known_args()