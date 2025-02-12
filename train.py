import torch
import torch.nn as nn

import configs as cfg
from model import CDC2F
from validate import validate

def train(model, train_loader, val_loader, criterion, optimizer, schedular, metrics, num_epoch, device, resume, logger, monitor):
    start_epoch = 0
    best_metric = 0
    best_epoch = 0
    if resume:
        checkpoint = torch.load(cfg.ckpt_save_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        schedular.load_state_dict(checkpoint['schedular_state'])
        start_epoch = checkpoint['cur_epoch']
        best_metric = checkpoint['best_score']
        best_epoch = checkpoint['best_epoch']
        logger.info(f'resume from epoch {start_epoch}')

    logger.info(model)

    for epoch in range(start_epoch, num_epoch):
        if isinstance(model, CDC2F):
            model.phase = 'train'
            model.fph.phase = 'train'
        model.train()
        if epoch >= 5:     # warm up stop, freeze bn
            freeze_batchnorm(model)
        for idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + idx
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.reshape(-1, 1, 256, 256)
            label = torch.where(label > 0, 1.0, 0)

            if isinstance(model, CDC2F):
                coarse_score, fre_score, fine_score_filtered = model(img1, img2)
                loss1 = criterion(torch.sigmoid(coarse_score), label)
                loss2 = criterion(torch.sigmoid(fine_score_filtered), label)
                loss3 = criterion(torch.sigmoid(fre_score), label)
                loss = loss1 + loss2 + loss3
                monitor.add_scalar('train/loss1', loss1, step)
                monitor.add_scalar('train/loss2', loss2, step)
                monitor.add_scalar('train/loss3', loss3, step)
            else :
                prob = model(img1, img2)
                loss = criterion(prob, label)
                monitor.add_scalar('train/loss', loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        schedular.step()
        # evaluate
        if isinstance(model, CDC2F):
            model.phase = 'val'
            model.fph.phase = 'val'
        logger.info(f'\ncurrent epoch {epoch}')
        current_metric, P, R = validate(model, val_loader, metrics, logger, None)
        if current_metric > best_metric:
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'schedular_state': schedular.state_dict(),
                    'cur_epoch': epoch,
                    'best_score': current_metric,
                    'best_epoch': best_epoch,
                }, cfg.ckpt_save_path)
            best_metric = current_metric
            best_epoch = epoch
        logger.info(f'best epoch: {best_epoch}')


def freeze_batchnorm(module):
    """
    冻结模型中的所有 BatchNorm 层。
    """
    for child in module.children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 冻结参数
            for param in child.parameters():
                param.requires_grad = False
            # 切换到 eval 模式，禁用统计量更新
            child.eval()
        else:
            # 递归处理子模块
            freeze_batchnorm(child)