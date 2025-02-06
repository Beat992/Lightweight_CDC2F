import torch
import configs as cfg
from validate import validate

def train(model, train_loader, val_loader, criterion, optimizer, metrics, num_epoch, device, logger, monitor):
    best_metric = 0
    best_epoch = 0
    model.train()
    logger.info(model)
    for epoch in range(num_epoch):
        model.phase = 'train'
        for idx, batch in enumerate(train_loader):
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.reshape(-1, 1, 256, 256)
            label = torch.where(label > 0, 1.0, 0)

            coarse_score, fre_score, fine_score_filtered, idx = model(img1, img2)
            loss1 = criterion(torch.sigmoid(coarse_score), label)
            loss2 = criterion(torch.sigmoid(fine_score_filtered), label)
            loss3 = criterion(torch.sigmoid(fre_score), label)
            loss = loss1 + loss2 + loss3
            # print(loss)
            monitor.add_scalar('train/loss1', loss1, epoch)
            monitor.add_scalar('train/loss2', loss2, epoch)
            monitor.add_scalar('train/loss3', loss3, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        model.phase = 'val'
        model.fph.phase = 'val'
        logger.info(f'\ncurrent epoch {epoch}')
        current_metric, P, R = validate(model, val_loader, metrics, logger, None)
        if current_metric > best_metric:
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'cur_epoch': epoch,
                    'best_score': current_metric
                }, cfg.ckpt_save_path)
            best_metric = current_metric
            best_epoch = epoch
        logger.info(f'best epoch: {best_epoch}')
