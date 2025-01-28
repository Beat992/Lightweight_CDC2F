import torch
import torch.nn as nn


class FreInfoExchange(nn.Module):
    def __init__(self, block_size, dct_size):
        super(FreInfoExchange, self).__init__()
        self.bands = dct_size ** 2
        self.patch_num = (block_size // dct_size) ** 2
        self.h = block_size // dct_size
        self.conv_toq = TwoLayerGroupConv(6, 16, 6, self.bands)
        self.conv_attn_cat = TwoLayerGroupConv(12, 32, 6, self.bands)
        self.conv_attn_sub = TwoLayerGroupConv(6, 16, 6, self.bands)
        self.conv_tov = TwoLayerGroupConv(6, 16, 6, self.bands)
        self.bs_block = 0

    def forward(self, x1, x2):
        self.bs_block = x1.shape[0]
        x1, x2 = self.transform(x1), self.transform(x2)

        x1, x2 = self.conv_toq(x1), self.conv_toq(x2) # 频段0: 6*h*w, 频段1: 6*h*w ...
        attn_cat = self.conv_attn_cat(   # 让x1和x2对应通道挨在一起然后分组卷积, 频段0: 6*2*h*w, 频段1 6*2*h*w
            torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)\
                .reshape(-1, 12 * self.bands, self.h, self.h))
        attn_sub = self.conv_attn_sub(x1 - x2)
        attn = torch.sigmoid(attn_cat + attn_sub)
        x1 = self.conv_tov(x1) * (attn + 1)
        x2 = self.conv_tov(x2) * (attn + 1)

        x1, x2 = self.transform_back(x1), self.transform_back(x2)
        return x1, x2

    def transform(self, x):
        # bs_block, p_n, 6, bands
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(self.bs_block, -1, self.h, self.h)
        return  x

    def transform_back(self, x):
        x = x.view(self.bs_block, self.bands, 6, self.patch_num)
        x = x.permute(0, 3, 2, 1)
        return x

class TwoLayerGroupConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bands):
        super(TwoLayerGroupConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*bands, mid_channels*bands, 3, 1, 1, groups=bands),
            nn.BatchNorm2d(mid_channels*bands),
            nn.ReLU(),
            nn.Conv2d(mid_channels*bands, out_channels*bands, 3, 1, 1, groups=bands)
        )
    def forward(self, x):
        return self.conv(x)