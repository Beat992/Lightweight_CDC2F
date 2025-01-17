import os

import torch
from tqdm import tqdm

from dataset import get_data_loader
from metrics import StreamSegMetrics
from metrics.Metric_SeK import metric_SeK
import numpy as np
import configs as cfg
from model import ResNetCD
from utils.logger import create_logger


def test():
    log = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                        net_name=cfg.model_version + cfg.backbone,
                        dataset_name=cfg.dataset_name,
                        phase='test')

    model = torch.load(os.path.join(cfg.base_path, f'prune/pruned_model_{cfg.backbone}.pth')).cuda()
    # model = ResNetCD(cfg.backbone).cuda()
    state_dict = torch.load(cfg.training_best_ckpt + '/LEVIR_CD_pruned_resnet502025-01-16-23-45-50_best.pth')
    model.load_state_dict(state_dict['model_state'])

    print(model)

    test_data_loader = get_data_loader(cfg.data_path, 'test', batch_size=cfg.val_batch_size, txt_path=cfg.test_txt_path)
    metrics = StreamSegMetrics(n_classes=2)
    pred_metric_all = []
    gt_metric_all = []

    metrics.reset()

    model.eval()

    with torch.no_grad():  # 修改
        for batch_idx, batch in enumerate(tqdm(test_data_loader)):
            inputs1, input2, mask = batch
            inputs1, inputs2, mask = inputs1.cuda(), input2.cuda(), mask.cuda()

            output = model(inputs1, inputs2)
            output = output.detach().cpu().numpy()
            pred_cm = np.uint8(np.where(output > 0.5, 1, 0))
            gt_cm = mask.detach().cpu().numpy().astype(np.uint8)
            gt_cm = np.where(gt_cm > 0, 1, 0)
            pred_metric_all.append(pred_cm[0])
            gt_metric_all.append(gt_cm[0])

            metrics.update(gt_cm, pred_cm)  # 添加

    # <-----返回F1、P、R、MIoU>
    F1, P, R = metric_SeK(infer_array=np.array(pred_metric_all), label_array=np.array(gt_metric_all), n_class=2,
                          log=log)

    score = metrics.get_results()  # 添加
    log.info('metric: ')
    log.info(metrics.to_str(score))

if __name__ == '__main__':
    test()