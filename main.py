import argparse
import configs as cfg
import numpy as np
import os
import random
import torch

from dataset import get_data_loader
from test import test
# from train_grad_accu import train
from train import train
from metrics import StreamSegMetrics
from model import CDC2F
from model.network_fre import PureFreCDNet
from utils.scheduler import warmup_cos_schedule
from utils.logger import create_logger
from utils.loss import compute_loss_CDC2F, compute_loss_PureFreCDNet
from utils.monitor import create_summery_writer


def setup_seed(seed):
    # 设置Python内置随机模块的种子
    random.seed(seed)
    # 设置numpy的种子
    np.random.seed(seed)
    # 设置PYTHONHASHSEED环境变量，确保hash函数的输出是确定的
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置PyTorch的种子
    torch.manual_seed(seed)
    # 如果使用GPU，也需要设置相应的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        # 禁用CUDNN的自动优化功能，以避免非确定性算法
        torch.backends.cudnn.benchmark = False
        # 强制CUDNN使用确定性算法
        torch.backends.cudnn.deterministic = True
        # 根据NVIDIA文档建议，设置CUBLAS_WORKSPACE_CONFIG环境变量
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        # 启用PyTorch的确定性算法选项
        # torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone, include resnet18/34/50/101, default: resnet50')
    parser.add_argument('--model_version', type=str, default='origin',
                        help='model version, origin / pruned / fre(default: origin)')
    parser.add_argument('--prune_strategy', type=str, default=None,
                        help='prune strategy, topk / mean')
    parser.add_argument('--prune_factor', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='levir-3167')
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()
    cfg.init(args)

    train_logger = create_logger(logger_name='train',
                                 logger_output_dir=os.path.join(cfg.base_path, 'log'),
                                 net_name=f'{args.model_version}_{args.backbone}',
                                 dataset_name=args.dataset,
                                 phase='train')
    train_logger.info(args)
    train_logger.info("save path: {}".format(cfg.ckpt_save_path))

    train_data_loader = get_data_loader(cfg.data_path[args.dataset], 'train', batch_size=args.batch_size, txt_path=cfg.train_txt_path)
    val_data_loader = get_data_loader(cfg.data_path[args.dataset], 'val', batch_size=args.batch_size, txt_path=cfg.val_txt_path, shuffle=False)
    test_data_loader = get_data_loader(cfg.data_path[args.dataset], 'test', batch_size=args.batch_size, txt_path=cfg.test_txt_path, shuffle=False)

    metrics = StreamSegMetrics(2)
    monitor = create_summery_writer(log_output_dir=os.path.join(cfg.base_path, 'monitor'),
                                    net_name=f'{args.model_version}_{args.backbone}',
                                    dataset_name=args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_version == 'origin':
        model = CDC2F(args.backbone, stages_num=5, phase='train', backbone_pretrained=True, dropout=args.dropout).to(device)
        # loss_func = compute_loss_CDC2F
    elif args.model_version == 'pruned' :
        model = torch.load(os.path.join(cfg.base_path, f'prune/pruned_{args.backbone}_{args.prune_strategy}.pth')).to(device)
        # loss_func = compute_loss_CDC2F
    else :
        model = PureFreCDNet(0.5, 'train', dropout=args.dropout).to(device)
        loss_func = compute_loss_PureFreCDNet
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCELoss()
    scheduler = warmup_cos_schedule(optimizer, 5, args.num_epochs, 5e-5)
    # train(model, train_data_loader, val_data_loader,
    #       criterion, optimizer, scheduler, metrics,
    #       args.num_epochs, 16, device, args.resume,
    #       train_logger, monitor)
    train(model, train_data_loader, val_data_loader,
          criterion, optimizer, scheduler, loss_func, metrics,
          args.num_epochs, device, args.resume,
          train_logger, monitor)
    test(model, test_data_loader, metrics, None, args)

