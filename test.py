import os
import sys
sys.path.append('..')
import configs as cfg
from model import CDC2F
from model.network_fre import PureFreCDNet
from validate import validate
from utils.logger import create_logger


def test(model, test_data_loader, metrics, writer, args):
    logger = create_logger(logger_name='test',
                           logger_output_dir=os.path.join(cfg.base_path, 'log'),
                           net_name=f'{args.model_version}_{args.backbone}',
                           dataset_name=args.dataset,
                           phase='test')
    if isinstance(model, CDC2F):
        model.phase = 'test'
        model.fph.phase = 'test'
    validate(model, test_data_loader, metrics, logger, writer)

if __name__ == '__main__':
    import argparse
    import torch
    from dataset import get_data_loader
    from metrics import StreamSegMetrics

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='backbone, include resnet18/34/50/101, default: resnet50')
    parser.add_argument('--model_version', type=str, default='origin',
                        help='model version, origin / pruned (default: origin)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='levir')
    args = parser.parse_args()
    cfg.init(args)
    # model = CDC2F(args.backbone, 5, 'test').cuda()
    model = PureFreCDNet(0.5, 'train', dropout=0.).cuda()
    # model = torch.load('./prune/pruned_resnet101_top70.pth')
    state_dict = torch.load(os.path.join('./checkpoints/levir-3167_fre_drop0_none_只第一个epoch更新bn.pth'))
    model.load_state_dict(state_dict['model_state'])
    test_data_loader = get_data_loader(cfg.data_path[args.dataset], 'test',
                                       batch_size=args.batch_size,
                                       txt_path=cfg.test_txt_path,
                                       drop_last=False,
                                       shuffle=True,)
    metrics = StreamSegMetrics(2)
    test(model, test_data_loader, metrics, None, args)