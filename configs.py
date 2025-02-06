import os

    # <--------path config--------->
base_path = '/home/zwb/prune_resnet'
data_path = {
    'levir-3167': '/home/pan2/zwb/LEVIR-CD-256-NEW', # 3167
    'levir': '/home/pan2/zwb/LEVIR_CD-256', # 3167
    'whu': '/home/pan2/zwb/WHU',
    'svcd': '/home/pan2/zwb/ChangeDetectionDataset'
}
ckpt_save_name = None
ckpt_save_path = None
train_txt_path = None
val_txt_path = None
test_txt_path = None
monitor_path = None

model_state_dict = {
    'resnet18': os.path.join(base_path, 'pretrained_weight/resnet18-5c106cde.pth'),
    'resnet34': os.path.join(base_path, 'pretrained_weight/resnet34-333f7ec4.pth'),
    'resnet50': os.path.join(base_path, 'pretrained_weight/resnet50-19c8e357.pth'),
    'resnet101': os.path.join(base_path, 'pretrained_weight/resnet101-5d3b4d8f.pth'),

}

def init(args):
    global ckpt_save_name, train_txt_path, val_txt_path, test_txt_path, ckpt_save_path, monitor_path

    ckpt_save_name = f"{args.dataset}_{args.model_version}_{args.backbone}.pth"
    train_txt_path = os.path.join(data_path[args.dataset], 'txt/train.txt')       # 修改
    val_txt_path = os.path.join(data_path[args.dataset], 'txt/val.txt')
    test_txt_path = os.path.join(data_path[args.dataset], 'txt/test.txt')

    training_best_ckpt = os.path.join(base_path, 'checkpoints')
    if not os.path.exists(training_best_ckpt):
        os.mkdir(training_best_ckpt)
    ckpt_save_path = os.path.join(training_best_ckpt, ckpt_save_name)
    cnt = 0
    while os.path.exists(ckpt_save_path):
        cnt += 1
        # 在原来的文件名后面加一个数字版本号
        ckpt_save_path = os.path.join(training_best_ckpt, ckpt_save_name.split('.')[0] + f'_v{cnt}' + '.pth')

    monitor_path = os.path.join(base_path, 'monitor')
    if not os.path.exists(monitor_path):
        os.mkdir(monitor_path)