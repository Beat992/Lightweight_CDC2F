import torch
from tqdm import tqdm
import torch.nn.functional as F
import configs as cfg
from validate import validate

def train(logger, train_loader, val_loader, model, criterion, optimizer, metrics, num_epoch, device):
    best_metric = 0
    model.train()
    for epoch in range(num_epoch):
        model.phase = 'train'
        for idx, batch in enumerate(tqdm(train_loader)):
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        current_metric, P, R = validate(model, val_loader, metrics, logger, epoch, None)
        if current_metric > best_metric:
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'cur_epoch': epoch,
                    'best_score': current_metric
                }, cfg.save_path)
            best_metric = current_metric
            best_epoch = epoch
            logger.info(f'best epoch: {best_epoch}')

def gt_filter(gt, idx, block_size):
    bs, ch, h, _ = gt.shape
    gt = F.unfold(gt, kernel_size=(block_size, block_size), padding=0, stride=(block_size, block_size))
    gt = gt.transpose(1, 2).reshape(-1, ch, block_size, block_size)
    gt = torch.index_select(gt, dim=0, index=idx)
    return gt
