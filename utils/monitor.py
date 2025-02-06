import sys
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def create_summery_writer(log_output_dir, net_name, dataset_name):
    root_output_dir = Path(log_output_dir)
    # <---------log_dir config---------->
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    dataset = dataset_name
    final_output_dir = root_output_dir / dataset / net_name
    if not final_output_dir.exists():
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

    # <---------log config--------->
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    monitor = SummaryWriter(log_dir=final_output_dir, comment=log_file,)

    return monitor