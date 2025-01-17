import os

# <---------usually modify----------->
prune = False
# prune = True
backbone = 'resnet50'
num_epochs = 200
train_batch_size = 16  # resnet18 32, 50: 8

model_version = "pruned" if prune else "origin"
dataset_name = 'LEVIR_CD'

base_save_name = f"{dataset_name}_{model_version}_{backbone}_best.pth"
save_name = base_save_name
counter = 1
while os.path.exists(save_name):
    save_name = f"{base_save_name.rsplit('.', 1)[0]}_{counter}.pth"
    counter += 1

# <---------usually modify----------->
channel_resnet18 = 1024
channel_resnet50 = 3904
weight_decay = [5e-5, 1e-2]
momentum = 0.90
val_batch_size = 1  # 设定的大了可能不能整除
lr_decay = False

t0_mean, t0_std = [0.46509804, 0.46042014, 0.39211731], [0.1854678, 0.17633663, 0.1648103]
t1_mean, t1_std = [0.36640143, 0.35843183, 0.31020384], [0.15798439, 0.15536726, 0.14649872]

# <--------path config--------->
base_path = '/home/zwb/prune_resnet'
data_path = '/home/pan2/zwb/LEVIR-CD-256-NEW'     # 3167
train_txt_path = os.path.join(data_path, 'txt/train.txt')       # 修改
val_txt_path = os.path.join(data_path, 'txt/val.txt')
# val_txt_path = os.path.join(data_path, 'txt_mini/val.txt')
test_txt_path = os.path.join(data_path, 'txt/test.txt')

training_best_ckpt = os.path.join(base_path, 'checkpoints')
if not os.path.exists(training_best_ckpt):
    os.mkdir(training_best_ckpt)
save_path = os.path.join(training_best_ckpt, save_name)

