import torch
from tqdm import tqdm

import configs as cfg
from validate import validate

def train(logger, train_loader, val_loader, model, criterion, optimizer, metrics, num_epoch, device):
    best_metric = 0
    for epoch in range(num_epoch):
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label = label.reshape(-1, 1, 256, 256)
            label = torch.where(label > 0, 1.0, 0)

            output = model(img1, img2)
            loss = criterion(output, label)
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




