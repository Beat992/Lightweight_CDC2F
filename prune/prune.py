import os
import re

import torch
from tqdm import tqdm
import configs as cfg
import torch_pruning as tp
from hook import *
from similarity import similarity_for_single_layer
from dataset import get_data_loader
from torchsummary import summary
from thop import profile
from model import CDC2F
resnet_blocks = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
}

def prune(model, layer_groups, prune_strategy='topk'):
    """
    :param layer_groups: 将所有卷积层分组，每组的格式为: layers-channle idx
    :param prune_strategy: 按topk剪枝还是按照平均值剪枝
    """
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()))
    for layer_group in layer_groups:
        for layer, channel in layer_group.items():
            prune_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=channel)
            print(prune_group)
            prune_group.prune()
    print(model)
    torch.save(model, f'pruned_{cfg.backbone}_{prune_strategy}.pth')

def params_flops(model):
    summary(model, [(3, 256, 256), (3, 256, 256)])
    flops, params = profile(model.cuda(), (torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()))
    print(f"模型的FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位显示
    print(f"模型的参数数量: {params / 1e6} M")
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))


if __name__ == '__main__':
    prune_strategy = 'topk'

    prune_factor1 = 0.3
    prune_factor2 = 0.3

    model = CDC2F(cfg.backbone, stages_num=5, phase='train', backbone_pretrained=True).eval().cuda()
    print(model)
    # params_flops(model)
    state_dict = torch.load(os.path.join(cfg.training_best_ckpt, 'LEVIR_CD_origin_resnet18_best.pth'))
    model.load_state_dict(state_dict['model_state'])
    val_loader = get_data_loader(cfg.data_path, 'val', 16, cfg.val_txt_path)
    register_hook(model)
    pattern1 = r'^resnet\.layer\d+\.\d+\.conv1$'
    pattern2 = r'^resnet\.layer\d+\.\d+\.conv2$'

    # pattern1 = r'^resnet\.layer\d+\.\d+\.conv[12]$'
    # pattern2 = r'^resnet\.layer\d+\.\d+\.conv3$'
    group_single = {}
    group_shortcut = {}
    for layer, name in layer_dict.items():
        if re.match(pattern1, name):
            group_single[name] = layer
        elif re.match(pattern2, name):
            group_shortcut[name] = layer

    print('----------------single layer----------------')
    for name, module in group_single.items():
        print(f"Layer: {name}, Module: {module}")

    print('----------------layer with shortcut----------------')
    for name, module in group_shortcut.items():
        print(f"Layer: {name}, Module: {module}")

    conv1_dist1 = {layer: [] for layer in group_single.keys()}
    conv2_dist2 = [[], [], [], []]
    model.zero_grad()
    for idx, batch in enumerate(tqdm(val_loader)):
        outputs_clear()
        img1, img2, _ = batch
        img1 = img1.cuda()
        img2 = img2.cuda()
        with torch.no_grad():
            model(img1, img2)

        for name, layer in group_single.items():
            dist = similarity_for_single_layer(layer_outputs[name][0], layer_outputs[name][1], 'euc')
            if len(conv1_dist1[name]) == 0:
                conv1_dist1[name] = dist
            else:
                conv1_dist1[name] = [a + b for a, b in zip(conv1_dist1[name], dist)]

        cur_layer = 0
        layer_list = list(group_shortcut.keys())
        for i in range(len(resnet_blocks[cfg.backbone])):  # [2, 2, 2, 2]
            end_layer = cur_layer + resnet_blocks[cfg.backbone][i]
            dists = []
            for cur_pos in range(cur_layer, end_layer):
                layer_name = layer_list[cur_pos]
                dist = similarity_for_single_layer(layer_outputs[layer_name][0], layer_outputs[layer_name][1], 'euc')
                dists.append(dist)
            dist_sum = [sum(x) for x in zip(*dists)]
            cur_layer = end_layer

            if len(conv2_dist2[i]) == 0:
                conv2_dist2[i] = dist_sum
            else:
                conv2_dist2[i] = [a + b for a, b in zip(conv2_dist2[i], dist_sum)]

    channel_idx1 = []
    for channel_dist in conv1_dist1.values():
        channel_dist = torch.tensor(channel_dist)
        if prune_strategy == 'topk':
            _, idx = torch.topk(channel_dist, round(len(channel_dist) * prune_factor1), largest=False)
        elif prune_strategy == 'std':
            mean = torch.mean(channel_dist)
            std = torch.std(channel_dist)
            idx = [i for i, x in enumerate(channel_dist) if x > mean + std]
        channel_idx1.append(idx.tolist())
    layer_channel1 = dict(zip([module for _, module in group_single.items()], channel_idx1))

    channel_idx2 = []
    for channel_dist in conv2_dist2:
        channel_dist = torch.tensor(channel_dist)
        if prune_strategy == 'topk':
            _, idx = torch.topk(channel_dist, round(len(channel_dist) * prune_factor2), largest=False)
        elif prune_strategy =='mean':
            mean = torch.mean(channel_dist)
            std = torch.std(channel_dist)
            idx = [i for i, x in enumerate(channel_dist) if x > mean + std]
        channel_idx2.append(idx.tolist())

    # pattern3 = r'^resnet\.layer\d+\.1\.conv3$'
    pattern3 = r'^resnet\.layer\d+\.1\.conv2$'
    conv2_group_ = []
    for name, module in group_shortcut.items():
        if re.match(pattern3, name):
            conv2_group_.append(module)
    layer_channel2 = dict(zip(conv2_group_, channel_idx2))

    remove_hook(handles)
    pruned_model = prune(model, [layer_channel1, layer_channel2], 'topk')

    # params_flops(pruned_model)