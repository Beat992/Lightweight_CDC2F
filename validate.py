import torch

from metrics.Metric_SeK import metric_SeK
import numpy as np

def validate(model, val_dataloader, metrics, log, epoch, writer):
    pred_metric_all = []
    gt_metric_all = []

    metrics.reset()

    model.eval()

    with torch.no_grad():  # 修改
        model.phase = 'val'
        model.fph.phase = 'val'
        for batch_idx, batch in enumerate(val_dataloader):
            inputs1, input2, mask = batch
            inputs1, inputs2, mask = inputs1.cuda(), input2.cuda(), mask.cuda()
            output = model(inputs1, inputs2)
            output = output[-1].detach().cpu().numpy()
            pred_cm = np.uint8(np.where(output > 0.5, 1, 0))
            gt_cm = mask.detach().cpu().numpy().astype(np.uint8)
            gt_cm = np.where(gt_cm > 0, 1, 0)
            pred_metric_all.append(pred_cm[0])
            gt_metric_all.append(gt_cm[0])

            metrics.update(gt_cm, pred_cm)  # 添加

    # <-----返回F1、P、R、MIoU>

    log.info(f'\ncurrent epoch {epoch}')
    F1, P, R = metric_SeK(infer_array=np.array(pred_metric_all), label_array=np.array(gt_metric_all), n_class=2,
                          log=log)


    score = metrics.get_results()  # 添加
    log.info('metric: ')
    log.info(metrics.to_str(score))

    return F1, P, R




