import argparse
import re

import torch_pruning as tp
from tqdm import tqdm

import configs as cfg
from dataset import get_data_loader
from hook import *
from model import CDC2F
from similarity import similarity_for_single_layer

resnet_blocks = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
}

def prune(model, layer_groups):
    """
    param: model: 待剪枝的模型
    param: layer_groups: 将所有卷积层分组，每组的格式为: layers-channel idx
    """
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()))
    for layer_group in layer_groups:
        for layer, channel in layer_group.items():
            prune_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=channel)
            print(prune_group)
            prune_group.prune()
    print(model)
    return model

def get_prune_idx(dists, prune_strategy, prune_factor=None):
    channel_idx = []
    for dist in dists:
        dist = torch.tensor(dist)
        idx = None
        if prune_strategy == 'topk':
            _, idx = torch.topk(dist, round(len(dist) * prune_factor), largest=False)
            idx = idx.tolist()
        elif prune_strategy == 'mean':
            mean = torch.mean(dist)
            idx = [i for i, x in enumerate(dist) if x > mean]
        channel_idx.append(idx)
    return channel_idx

if __name__ == '__main__':
    # ----- args parser -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone, include resnet18/34/50/101, default: resnet50')
    parser.add_argument('--dataset', type=str, default='levir')
    parser.add_argument('--prune_strategy', type=str, default='topk',
                        help='prune strategy, topk / mean')
    parser.add_argument('--sim_type', type=str, default='euc')
    parser.add_argument('--prune_factor', type=float, default=0.5)
    parser.add_argument('--pth_path', type=str)
    args = parser.parse_args()

    # ----- model -----
    model = CDC2F(args.backbone, stages_num=5, phase='val', backbone_pretrained=True).eval().cuda()
    print(model)
    state_dict = torch.load(args.pth_path)
    model.load_state_dict(state_dict['model_state'])
    register_hook(model)

    # ----- dataset -----
    val_loader = get_data_loader(cfg.data_path[args.dataset], 'val', 2, cfg.val_txt_path, drop_last=False)

    # ----- prepare layers -----
    if args.backbone in ['resnet18', 'resnet34']:
        single_layer = r'^resnet\.layer\d+\.\d+\.conv1$'
        shortcut_layer = r'^resnet\.layer\d+\.\d+\.conv2$'
        shortcut_layer_last = r'^resnet\.layer\d+\.1\.conv2$'
    else:
        single_layer = r'^resnet\.layer\d+\.\d+\.conv[12]$'
        shortcut_layer = r'^resnet\.layer\d+\.\d+\.conv3$'
        shortcut_layer_last = r'^resnet\.layer\d+\.1\.conv3$'

    group_single = {}
    group_shortcut = {}
    for layer, name in layer_dict.items():
        if re.match(single_layer, name):
            group_single[name] = layer
        elif re.match(shortcut_layer, name):
            group_shortcut[name] = layer

    print('----------------single layer----------------')
    for name, module in group_single.items():
        print(f"Layer: {name}, Module: {module}")

    print('----------------layer with shortcut----------------')
    for name, module in group_shortcut.items():
        print(f"Layer: {name}, Module: {module}")

    # ----- inference and calculate similarity -----
    dist_single = {layer: [] for layer in group_single.keys()}
    dist_shortcut = [[], [], [], []]
    model.zero_grad()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            outputs_clear()
            img1, img2, _ = batch
            img1 = img1.cuda()
            img2 = img2.cuda()

            model(img1, img2)

            # ----- dist of single layer -----
            for name, layer in group_single.items():
                dist = similarity_for_single_layer(layer_outputs[name][0], layer_outputs[name][1], args.sim_type)
                if len(dist_single[name]) == 0:
                    dist_single[name] = dist
                else:
                    dist_single[name] = [a + b for a, b in zip(dist_single[name], dist)]

            # ----- dist of layer with shortcut for each stage-----
            cur_layer = 0
            layer_list = list(group_shortcut.keys())
            for stage in range(4):  # eg. [2, 2, 2, 2]
                end_layer = cur_layer + resnet_blocks[args.backbone][stage]
                dists = []
                for cur_pos in range(cur_layer, end_layer):
                    layer_name = layer_list[cur_pos]
                    dist = similarity_for_single_layer(layer_outputs[layer_name][0], layer_outputs[layer_name][1], args.sim_type)
                    dists.append(dist)
                dist_sum = [sum(x) for x in zip(*dists)]
                cur_layer = end_layer
                if len(dist_shortcut[stage]) == 0:
                    dist_shortcut[stage] = dist_sum
                else:
                    dist_shortcut[stage] = [a + b for a, b in zip(dist_shortcut[stage], dist_sum)]

    # ----- get prune idx for each layer -----
    channel_idx1 = get_prune_idx(dist_single.values(), args.prune_strategy, args.prune_factor)
    layer_channel1 = dict(zip([module for _, module in group_single.items()], channel_idx1))

    channel_idx2 = get_prune_idx(dist_shortcut, args.prune_strategy, args.prune_factor)
    shortcut_prune_group = []
    for name, module in group_shortcut.items():
        if re.match(shortcut_layer_last, name):
            shortcut_prune_group.append(module)
    layer_channel2 = dict(zip(shortcut_prune_group, channel_idx2))

    remove_hook(handles) # remove hook, or the next inference will have problem

    # ----- prune -----
    pruned_model = prune(model, [layer_channel1, layer_channel2])
    if args.prune_strategy == 'topk':
        pruned_model_name =  f'pruned_{args.backbone}_{args.sim_type}_top{int(args.prune_factor*100)}_{args.dataset}.pth'
    else :
        pruned_model_name = f'pruned_{args.backbone}_{args.sim_type}_mean_{args.dataset}.pth'
    torch.save(pruned_model, pruned_model_name)
    # params_flops(pruned_model)