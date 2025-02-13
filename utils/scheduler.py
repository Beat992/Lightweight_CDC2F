import math
from torch.optim.lr_scheduler import LambdaLR
def warmup_cos_schedule(optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
    """
    自定义 Warm-up 调度器。
    :param optimizer: 优化器
    :param warmup_epochs: Warm-up 的epoch数
    :param total_epochs: 总epoch数
    """

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # 线性增长到初始学习率
            return float(current_epoch) + 1 / float(max(1, warmup_epochs))
        else:
            # 按余弦退火或其他方式衰减
            return max(min_lr, 0.5 * (
                        1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs))))

    return LambdaLR(optimizer, lr_lambda)