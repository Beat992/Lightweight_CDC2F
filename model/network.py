import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones import resnet
import configs as cfg

model_state_dict = {
    'resnet18': os.path.join(cfg.base_path, 'pretrained_weight/resnet18-5c106cde.pth'),
    'resnet34': os.path.join(cfg.base_path, 'pretrained_weight/resnet34-333f7ec4.pth'),
    'resnet50': os.path.join(cfg.base_path, 'pretrained_weight/resnet50-19c8e357.pth'),
}

backbone_total_channels = {
    'resnet18': 1024,
    'resnet34': 1024,
    'resnet50': 3904,
}

class ResNetCD(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(ResNetCD, self).__init__()
        self.backbone = backbone
        if backbone == 'resnet18':
            self.resnet = resnet.resnet18()
        elif backbone == 'resnet34':
            self.resnet = resnet.resnet34()
        elif backbone == 'resnet50':
            self.resnet = resnet.resnet50()
        # self.load_pretrained_backbone()

        self.dim_reduction_conv = nn.Sequential(
            # 如果用resnet50，融合通道的时候应该平滑一点，分几次降低到最低的通道数
            nn.Conv2d(in_channels=backbone_total_channels[self.backbone], out_channels=256, kernel_size=1, stride=1, padding=0, ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )

        self.pred_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward_resnet(self, x):
        conv1_feature = self.resnet.conv1(x)
        conv1_feature = self.resnet.bn1(conv1_feature)
        conv1_feature = self.resnet.relu(conv1_feature)
        conv2_feature = self.resnet.maxpool(conv1_feature)  # 1/2, out_channel=64

        conv2_feature = self.resnet.layer1(conv2_feature)  # 1/4, in=64, out=64

        conv3_feature = self.resnet.layer2(conv2_feature)  # 1/8, in=64, out=128

        conv4_feature = self.resnet.layer3(conv3_feature)  # 1/8, in=128, out=256

        conv5_feature = self.resnet.layer4(conv4_feature)

        return [conv1_feature, conv2_feature, conv3_feature, conv4_feature, conv5_feature]
        # return conv5_feature

    def feature_extraction(self, x):
        x = self.forward_resnet(x)
        x = self.upsample_and_cat(x)
        x = self.dim_reduction_conv(x)

        return x

    def upsample_and_cat(self, x):
        size = x[0].shape[-1]
        output = x[0].clone().requires_grad_(True)
        for feature in x[1:]:
            feature = F.interpolate(feature, [size, size], mode='bilinear', align_corners=True)
            output = torch.cat([output, feature], dim=1)
        return output

    def load_pretrained_backbone(self):
        state_dict = torch.load(model_state_dict[self.backbone])
        self.resnet.load_state_dict(state_dict)

    def forward(self, x0, x1):
        x0 = self.feature_extraction(x0)
        x1 = self.feature_extraction(x1)
        # x0 = F.interpolate(x0, [128, 128], mode='bilinear', align_corners=True)
        # x1 = F.interpolate(x1, [128, 128], mode='bilinear', align_corners=True)
        change_score = torch.abs(x0 - x1)
        change_score = self.pred_head(change_score)

        return torch.sigmoid(change_score)

# print(ResNetCD('resnet18'))