import os
import configs as cfg
import torch
from dataset import get_data_loader
from train import train
from metrics import StreamSegMetrics
from model import ResNetCD
from utils.logger import create_logger

if __name__ == '__main__':
    logger = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                        net_name=cfg.model_version + cfg.backbone,
                        dataset_name=cfg.dataset_name,
                        phase='train')
    train_data_loader = get_data_loader(cfg.data_path, 'train', batch_size=cfg.train_batch_size, txt_path=cfg.train_txt_path)
    # val_data_loader = get_data_loader(cfg.data_path, 'train', batch_size=cfg.train_batch_size, txt_path=cfg.train_txt_path)
    val_data_loader = get_data_loader(cfg.data_path, 'val', batch_size=cfg.val_batch_size, txt_path=cfg.val_txt_path)
    metrics = StreamSegMetrics(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = ResNetCD(cfg.backbone).to(device)
    model = torch.load(os.path.join(cfg.base_path, f'prune/pruned_model_{cfg.backbone}.pth')).cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    train(logger, train_data_loader, val_data_loader, model, criterion, optimizer, metrics, cfg.num_epochs, device)