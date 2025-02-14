import os
import configs as cfg
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loader
from metrics import StreamSegMetrics, metric_SeK
from model import CDC2F
from model.network_fre import PureFreCDNet
from utils.logger import create_logger
from utils.loss import compute_loss_CDC2F, compute_loss_PureFreCDNet
from utils.monitor import create_summery_writer
from utils.scheduler import warmup_cos_schedule


class Trainer:
    def __init__(self, args, model, is_grad_accum=False, accumulation_steps=None):
        self.backbone = args.backbone
        self.version = args.version
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.wd
        self.dropout = args.dropout
        self.dataset = args.dataset
        self.isResume = args.isResume
        self.prune_strategy = args.prune_strategy
        self.prune_freq = args.prune_freq
        self.is_grad_accum = is_grad_accum
        self.accumulation_steps = accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.init(args)
        self.metrics = StreamSegMetrics(2)
        self.model = self.build_model() if model is None else model
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.loss_func = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.build_trainer()
        self.train_logger = create_logger(logger_name='train',
                                 logger_output_dir=os.path.join(cfg.base_path, 'log'),
                                 net_name=f'{args.model_version}_{args.backbone}',
                                 dataset_name=args.dataset,
                                 phase='train')
        self.monitor = create_summery_writer(log_output_dir=os.path.join(cfg.base_path, 'monitor'),
                                    net_name=f'{args.model_version}_{args.backbone}',
                                    dataset_name=args.dataset)
        self.best_metric = 0.
        self.best_epoch = 0
        self.start_epoch = 0
    def train(self):
        if self.isResume:
            self.resume()
        self.train_logger.info(self.model)

        for epoch in range(self.start_epoch, self.num_epochs):
            if isinstance(self.model, CDC2F):
                self.model.phase = 'train'
                self.model.fph.phase = 'train'

            self.model.train()

            if epoch >= 5:     # warm up stop, freeze bn
                self.freeze_bn(self.model)

            for idx, batch in enumerate(self.train_loader):
                step = epoch * len(self.train_loader) + idx
                img1, img2, label = batch
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                label = label.reshape(-1, 1, 256, 256)
                label = torch.where(label > 0, 1.0, 0)
                loss = self.loss_func(self.model, self.criterion, img1, img2, label, self.monitor, step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_logger.info(f'\ncurrent epoch {epoch}')
            current_metric, P, R = self.validate()
            self.save_model(current_metric, epoch)

    def train_with_ga(self):
        if self.isResume:
            self.resume()

        self.train_logger.info(self.model)

        for epoch in range(self.start_epoch, self.num_epochs):
            if isinstance(self.model, CDC2F):
                self.model.phase = 'train'
                self.model.fph.phase = 'train'
            self.model.train()
            # if epoch >= 5:     # warm up stop, freeze bn
            #     freeze_bn(model)

            cnt = 0
            batch_loss = []
            for idx, batch in enumerate(self.train_loader):
                cnt += 1
                step = epoch * len(self.train_loader) + idx
                img1, img2, label = batch
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                label = label.reshape(-1, 1, 256, 256)
                label = torch.where(label > 0, 1.0, 0)

                prob = self.model(img1, img2)
                loss = self.criterion(prob, label)
                batch_loss.append(loss)

                loss.backward()
                if cnt == self.accumulation_steps:
                    self.optimizer.step()  # 更新参数
                    self.optimizer.zero_grad()  # 清空梯度
                    self.monitor.add_scalar('train/loss1', sum(batch_loss) / self.accumulation_steps, step // self.accumulation_steps)
                    batch_loss.clear()
                    cnt = 0

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_logger.info(f'\ncurrent epoch {epoch}')
            current_metric, P, R = self.validate()
            self.save_model(current_metric, epoch)

    def validate(self):
        # evaluate
        if isinstance(self.model, CDC2F):
            self.model.phase = 'val'
            self.model.fph.phase = 'val'
        pred_all = []
        gt_all = []
        self.metrics.reset()
        self.model.eval()
        with torch.no_grad():  # 修改
            for batch_idx, batch in enumerate(self.val_loader):
                inputs1, input2, mask = batch
                inputs1, inputs2, mask = inputs1.cuda(), input2.cuda(), mask
                output = self.model(inputs1, inputs2)
                pred_cm = torch.where(output > 0.5, 1, 0).to(torch.uint8).cuda()
                gt_cm = torch.where(mask > 0, 1, 0).to(torch.uint8).cuda()
                pred_all.append(pred_cm)
                gt_all.append(gt_cm)

                self.metrics.update(gt_cm, pred_cm)  # 添加

        # <-----返回F1、P、R、MIoU>
        pred_all = torch.cat(pred_all, dim=0)
        gt_all = torch.cat(gt_all, dim=0)
        F1, P, R = metric_SeK(infer_array=pred_all, label_array=gt_all, n_class=2,
                              log=self.train_logger)
        score = self.metrics.get_results()  # 添加
        self.train_logger.info('metric: ')
        self.train_logger.info(self.metrics.to_str(score))

        return F1, P, R

    def build_model(self):
        if self.version == 'origin':
            self.model = CDC2F(self.backbone, stages_num=5, phase='train', backbone_pretrained=True,
                          dropout=self.dropout).to(self.device)
        elif self.version == 'pruned':
            self.model = torch.load(
                os.path.join(cfg.base_path, f'prune/pruned_{self.backbone}_{self.prune_strategy}.pth')).to(self.device)
        else:
            self.model = PureFreCDNet(0.5, 'train', dropout=self.dropout).to(self.device)


    def build_optimizer(self):
        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def build_scheduler(self):
        self.scheduler = warmup_cos_schedule(self.optimizer, 5, self.num_epochs, 5e-5)

    def build_loss(self):
        self.criterion = nn.BCELoss()
        if not self.is_grad_accum:
            if self.version == ('origin', 'pruned'):
                self.loss_func = compute_loss_CDC2F
            elif self.version == 'fre':
                self.loss_func = compute_loss_PureFreCDNet

    def build_dataloader(self):
        self.train_loader = get_data_loader(cfg.data_path[self.dataset], 'train', batch_size=self.batch_size,
                                            txt_path=cfg.train_txt_path)
        self.val_loader = get_data_loader(cfg.data_path[self.dataset], 'val', batch_size=self.batch_size,
                                          txt_path=cfg.val_txt_path, shuffle=False)
        self.test_loader = get_data_loader(cfg.data_path[self.dataset], 'test', batch_size=self.batch_size,
                                           txt_path=cfg.test_txt_path, shuffle=False)

    def freeze_bn(self, module):
        """
        冻结模型中的所有 BatchNorm 层。
        """
        for child in module.children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 冻结参数
                for param in child.parameters():
                    param.requires_grad = False
                # 切换到 eval 模式，禁用统计量更新
                child.eval()
            else:
                # 递归处理子模块

               self.freeze_bn(child)

    def build_trainer(self):
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss()
        self.build_dataloader()

    def save_model(self, current_metric, epoch):
        if current_metric > self.best_metric:
            torch.save(
                {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'schedular_state': self.scheduler.state_dict(),
                    'cur_epoch': epoch,
                    'best_score': current_metric,
                }, cfg.ckpt_save_path)
            self.best_metric = current_metric
            self.best_epoch = epoch
        self.train_logger.info(f'best epoch: {self.best_epoch}')

    def resume(self):
        checkpoint = torch.load(cfg.ckpt_save_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['schedular_state'])
        self.start_epoch = checkpoint['cur_epoch']
        self.best_metric = checkpoint['best_score']
        self.best_epoch = checkpoint['cur_epoch']
        self.train_logger.info(f'resume from epoch {self.start_epoch}')