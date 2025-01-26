import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaFreInteractionFusion(nn.Module):
    """
        func: Interacting and fusing frequency  and spatial features
        input: spatial features: n, 64, block_size, block_size
               frequency features: n, 6, block_size, block_size
        output: fused features: n, 32, block_size, block_size
    """

    def __init__(self, block_size):
        super(SpaFreInteractionFusion, self).__init__()
        self.block_size = block_size
        self.conv_channel_align_spa = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        self.conv_channel_align_fre = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        )

        self.conv_attn_self_domain = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )

        self.conv_attn_cross_domain = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )
        self.conv_attn_mc_guided = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )
        self.conv_cross_add = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
        )

    def _forward_single_domain(self, fs, fc, mc):
        attn_cross = self.attn_cross_domain(fc)
        # attn_cross = torch.sigmoid(self.conv_prob_attn(torch.cat([mc, attn_cross], dim=1)))
        attn_mc_guided_cross = torch.sigmoid(self.attn_mc_guided(mc + attn_cross))
        attn_self = torch.sigmoid(self.conv_attn_self_domain(fs))
        fs = fs * (1 + 0.7 * attn_self + 0.3 * attn_mc_guided_cross)
        fc = self.conv_cross_add(fc)

        return fs + fc

    def forward(self, X_spa, X_fre, mask_coarse):
        mc = mask_coarse.unsqueeze(1)
        mc = F.interpolate(mc, self.block_size)

        X_spa = self.conv_channel_align_spa(X_spa)
        X_fre = self.conv_channel_align_fre(X_fre)

        X_spa = self._forward_single_domain(X_spa, X_fre, mc)
        X_fre = self._forward_single_domain(X_fre, X_spa, mc)
        return torch.cat([X_spa, X_fre], dim=1)
