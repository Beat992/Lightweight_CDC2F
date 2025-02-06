import argparse
import os
import configs as cfg
import torch
from dataset import get_data_loader
from test import test

from train import train
from metrics import StreamSegMetrics
from model import CDC2F
from utils.logger import create_logger
from utils.monitor import create_summery_writer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone, include resnet18/34/50/101, default: resnet50')
    parser.add_argument('--model_version', type=str, default='origin',
                        help='model version, origin / pruned (default: origin)')
    parser.add_argument('--prune_strategy', type=str, default=None,
                        help='prune strategy, topk / mean&std')
    parser.add_argument('--prune_factor', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='LEVIR_CD')
    args = parser.parse_args()
    cfg.init(args)

    train_logger = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                                 net_name=f'{args.model_version}_{args.backbone}',
                                 dataset_name=args.dataset,
                                 phase='train')
    train_logger.info(args)
    train_logger.info("save path: {}".format(cfg.ckpt_save_path))

    train_data_loader = get_data_loader(cfg.data_path, 'train', batch_size=args.batch_size, txt_path=cfg.train_txt_path)
    val_data_loader = get_data_loader(cfg.data_path, 'val', batch_size=args.batch_size, txt_path=cfg.val_txt_path, shuffle=False)
    test_data_loader = get_data_loader(cfg.data_path, 'test', batch_size=args.batch_size, txt_path=cfg.test_txt_path)

    metrics = StreamSegMetrics(2)
    monitor = create_summery_writer(log_output_dir=os.path.join(cfg.base_path, 'monitor'),
                                    net_name=f'{args.model_version}_{args.backbone}',
                                    dataset_name=args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_version == 'origin':
        model = CDC2F(args.backbone, stages_num=5, phase='train', backbone_pretrained=True).to(device)
    else :
        model = torch.load(os.path.join(cfg.base_path, f'prune/pruned_{args.backbone}_{args.prune_strategy}.pth')).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    train(model, train_data_loader, val_data_loader, criterion, optimizer, metrics, args.num_epochs, device, train_logger, monitor)
    monitor.close()
    test(model, test_data_loader, metrics, None, args)

