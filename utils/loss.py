import torch
def compute_loss_CDC2F(model, criterion, img1, img2, label, monitor, step):
    coarse_score, fre_score, fine_score_filtered = model(img1, img2)
    loss1 = criterion(torch.sigmoid(coarse_score), label)
    loss2 = criterion(torch.sigmoid(fine_score_filtered), label)
    loss3 = criterion(torch.sigmoid(fre_score), label)
    loss = loss1 + loss2 + loss3
    monitor.add_scalar('train/loss1', loss1, step)
    monitor.add_scalar('train/loss2', loss2, step)
    monitor.add_scalar('train/loss3', loss3, step)
    return loss

def compute_loss_PureFreCDNet(model, criterion, img1, img2, label, monitor, step):
    prob = model(img1, img2)
    loss = criterion(prob, label)
    monitor.add_scalar('train/loss', loss, step)
    return loss