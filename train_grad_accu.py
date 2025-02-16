import torch
import configs as cfg
from model import CDC2F
from validate import validate

def train_with_ga(model, train_loader, val_loader,
          criterion, optimizer, schedular, metrics,
          num_epoch, accumulation_steps, device, resume,
          logger, monitor):
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
        best_epoch = checkpoint['cur_epoch']
        logger.info(f'resume from epoch {start_epoch}')

    logger.info(model)

    for epoch in range(start_epoch, num_epoch):
        if isinstance(model, CDC2F):
            model.phase = 'train'
            model.fph.phase = 'train'
        model.train()
        # if epoch >= 5:     # warm up stop, freeze bn
        #     freeze_bn(model)

        cnt = 0
        loss_1 = []
        for idx, batch in enumerate(train_loader):
            cnt += 1
            step = epoch * len(train_loader) + idx
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.reshape(-1, 1, 256, 256)
            label = torch.where(label > 0, 1.0, 0)

            prob = model(img1, img2)
            loss1 = criterion(prob, label)
            loss_1.append(loss1)

            loss1.backward()
            if cnt == accumulation_steps:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清空梯度
                monitor.add_scalar('train/loss1', sum(loss_1) / accumulation_steps, step // accumulation_steps)

                loss_1.clear()

                cnt = 0

        if schedular is not None:
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
                    'schedular_state': schedular.state_dict() if schedular is not None else None,
                    'cur_epoch': epoch,
                    'best_score': current_metric,
                }, cfg.ckpt_save_path)
            best_metric = current_metric
            best_epoch = epoch
        logger.info(f'best epoch: {best_epoch}')