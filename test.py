import os
import configs as cfg
from validate import validate
from utils.logger import create_logger


def test(model, test_data_loader, metrics, writer, args):
    logger = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                           net_name=f'{args.model_version}_{args.backbone}',
                           dataset_name=args.dataset,
                           phase='test')
    logger.info(model)
    model.phase = 'test'
    model.fph.phase = 'test'
    validate(model, test_data_loader, metrics, logger, writer)
