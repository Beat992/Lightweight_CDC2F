import os

    # <--------path config--------->
base_path = '/home/zwb/prune_resnet'
data_path = {
    'levir3167': '/home/pan2/zwb/LEVIR-CD-256-NEW', # 3167
    'levir': '/home/pan2/zwb/LEVIR_CD-256', # 3167
    'whu': '/home/pan2/zwb/WHU',
    'svcd': '/home/pan2/zwb/ChangeDetectionDataset'
}
ckpt_save_name = None
ckpt_save_path = None
train_txt_path = None
val_txt_path = None
test_txt_path = None


def init(args):
    global ckpt_save_name, train_txt_path, val_txt_path, test_txt_path, ckpt_save_path

    ckpt_save_name = f"{args.dataset}_{args.model_version}_{args.backbone}.pth"

    train_txt_path = os.path.join(data_path[args.dataset], 'txt/train.txt')       # 修改
    val_txt_path = os.path.join(data_path[args.dataset], 'txt/val.txt')
    test_txt_path = os.path.join(data_path[args.dataset], 'txt/test.txt')

    training_best_ckpt = os.path.join(base_path, 'checkpoints')
    if not os.path.exists(training_best_ckpt):
        os.mkdir(training_best_ckpt)
    ckpt_save_path = os.path.join(training_best_ckpt, ckpt_save_name)

