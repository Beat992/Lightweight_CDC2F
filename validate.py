import torch

from metrics.Metric_SeK import metric_SeK
import numpy as np

def validate(model, val_dataloader, metrics, log, writer):
    pred_all = []
    gt_all = []
    metrics.reset()
    model.eval()
    with torch.no_grad():  # 修改
        for batch_idx, batch in enumerate(val_dataloader):
            inputs1, input2, mask = batch
            inputs1, inputs2, mask = inputs1.cuda(), input2.cuda(), mask
            output = model(inputs1, inputs2)
            pred_cm = torch.where(output > 0.5, 1, 0).to(torch.uint8).cuda()
            gt_cm = torch.where(mask > 0, 1, 0).to(torch.uint8).cuda()
            pred_all.append(pred_cm)
            gt_all.append(gt_cm)

            metrics.update(gt_cm, pred_cm)  # 添加

    # <-----返回F1、P、R、MIoU>
    pred_all = torch.cat(pred_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    F1, P, R = metric_SeK(infer_array=pred_all, label_array=gt_all, n_class=2,
                          log=log)
    score = metrics.get_results()  # 添加
    log.info('metric: ')
    log.info(metrics.to_str(score))

    return F1, P, R




