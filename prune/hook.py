import torch
import torch.nn as nn
from model.module_coarse_detection import CoarseDetection as ResNetCD

layer_dict = {}   # key: module, value: layer_name
layer_outputs = {}  # key: layer_name, value: output
handles = []

def save_intermediate_outputs(module, input, output):
    layer_name = layer_dict[module]
    layer_outputs[layer_name].append(output)


def register_hook(model):
    # 遍历模型中的每个子模块，找到所有卷积层并注册钩子
    for name, module in model.coarse_detection.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_dict[module] = name
            layer_outputs[name] = []
            handles.append(module.register_forward_hook(save_intermediate_outputs))

def remove_hook(handles):
    for handle in handles:
        handle.remove()

def outputs_clear():
    # 清空layer_outputs中所有的值
    for key in layer_outputs.keys():
        layer_outputs[key] = []

if __name__ == '__main__':
    model = ResNetCD('resnet18')
    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)
    register_hook(model)

    output = model(input1, input2)

    # 遍历layer_dict字典，打印每个卷积层的名称
    print('---------------卷积层---------------')
    for layer, name in layer_dict.items():
        print(f"Layer: {name}, Module: {layer}")
    print('---------------卷积层输出---------------')
    # 遍历layer_outputs字典，打印每个层的输出
    for layer, outputs in layer_outputs.items():
        print(f"Layer: {layer}")
        print(outputs[0].shape)
        if len(outputs) > 1:
            print(outputs[1].shape)

