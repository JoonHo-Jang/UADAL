
#! /bin/bash


### UADAL ### NeurIPS 2022 ###

# 1. please download the datasets as we introduce in appendix file.
# 2. make a directory '../datasets', and put each dataset into the directory.
# 3. (optional) implement "python3 utils/list_xxx.py" which xxx stands for the name of datasets.
# 4. implement below scripts.


python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "A" --target_domain "W" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "A" --target_domain "D" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "D" --target_domain "W" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "W" --target_domain "D" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "D" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "office" --source_domain "W" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0


python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "A" --target_domain "W" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "A" --target_domain "D" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "D" --target_domain "W" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "W" --target_domain "D" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "D" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "office" --source_domain "W" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0


python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "UADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0


python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "A" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "C" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "P" --target_domain "R" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "A" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "C" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
python3.8 train.py --model "cUADAL" --net 'resnet50' --dataset "officehome" --source_domain "R" --target_domain "P" --warmup_iter 2000 --training_iter 100 --update_term 10 --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
