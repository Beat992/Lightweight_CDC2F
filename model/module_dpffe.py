from math import sqrt
import torch
import torch.nn as nn
from model.layers import Transformer, TransformerDecoder

'''
    input: DCT patches which need to refinement, some of which is 0,
           means they are either change completely or no change 
    shape: (bs*block_num) * patch_num * 3 * bands
    output: 
'''


class DualPerFreFeatureExtractor(nn.Module):
    def __init__(self,
                 dct_size=4,
                 block_size=64,
                 encoder_depth=1,
                 encoder_heads=8,
                 encoder_dim=8,
                 decoder_depth=1,
                 decoder_heads=4,
                 decoder_dim=8,
                 dropout=0.5):
        super(DualPerFreFeatureExtractor, self).__init__()
        self.dct_size = dct_size
        self.block_size = block_size
        self.bands = dct_size ** 2
        self.patch_num = (block_size // dct_size) ** 2

        self.band_group_conv0 = BandGroupConv(self.bands * 3, 16, 8, 1)
        self.band_group_conv1 = BandGroupConv(self.bands * 3, 4, 32, 1)
        self.band_group_conv2 = BandGroupConv(self.bands * 3, 1, 128, 1)
        self.seblock = SEBlock(channel=self.bands * 3 * 2, reduction=6)

        self.dc_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        len_token = 3 * self.bands
        self.layer_embedding = nn.Linear(len_token, len_token)
        self.pos_embedding_enc = nn.Parameter(torch.randn(1, self.patch_num, len_token))
        self.encoder = Transformer(dim=len_token, depth=encoder_depth, heads=encoder_heads,
                                   dim_head=encoder_dim, mlp_dim=len_token, dropout=dropout)
        self.pos_embedding_dec = nn.Parameter(torch.randn(1, self.patch_num, len_token))
        self.decoder = TransformerDecoder(dim=len_token, depth=decoder_depth, heads=decoder_heads,
                                            dim_head=decoder_dim, dropout=dropout, mlp_dim=decoder_dim)


        self.transform_statics = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.transform_patch = nn.Sequential(
            nn.Linear(self.bands, 64),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.patch_attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm2d(self.patch_num),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _band_level(self, x):
        bs_block, patch_num, ch, bands = x.shape
        h = self.block_size // self.dct_size
        x = x.transpose(1, 3).reshape(bs_block, -1, h, h)
        x = self.band_group_conv0(x) + self.band_group_conv1(x) + self.band_group_conv2(x) + x
        x = self.seblock(x)
        x = x.reshape(bs_block, bands, ch, patch_num).transpose(1, 3)
        return x

    def _patch_level(self, x):
        h = self.block_size // self.dct_size
        dc = x[:, :, :, 0].transpose(1, 2)
        dc = dc.reshape(x.shape[0], 3, h, h)
        dc = self.dc_conv(dc)
        dc = dc.reshape(x.shape[0], 3, self.patch_num).transpose(1, 2)

        x = self._patch_attention(x)
        x = self._forward_transformer(x)
        x[:, :, :, 0] = x[:, :, :, 0] + dc
        return x

    def _patch_attention(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        min_vals = x.min(dim=-1, keepdim=True)[0]
        max_vals = x.max(dim=-1, keepdim=True)[0]
        peak_frequency = torch.argmax(x, dim=-1, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True))
        energy = torch.sum(torch.square(x), dim=-1, keepdim=True)
        # st = torch.cat([var, mean, max_vals, min_vals], dim=-1)
        st = torch.cat([var, mean, max_vals, min_vals, peak_frequency, rms, energy], dim=-1)
        st = self.transform_statics(st)
        x_ = self.transform_patch(x)
        attn = self.patch_attn(st + x_)
        x = attn * x
        return x

    def _forward_transformer(self, x):
        bs_block, patch_num, ch, bands = x.shape
        x = x.reshape(bs_block, patch_num, -1)
        m = torch.clone(x)
        x = self.embedding_layer(x)
        x += self.pos_embedding
        x = self.encoder(x)
        m += self.pos_embedding_dec
        x = self.decoder(m, x)
        x = x.view(bs_block, patch_num, ch, -1)
        return x

    def _forward_single(self, x):
        patch_level = self._patch_level(x)
        band_level = self._band_level(x)
        return torch.cat([patch_level, band_level], dim=2)

    def forward(self, t0, t1):
        t0 = self._forward_single(t0)
        t1 = self._forward_single(t1)
        return t0, t1

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channel)
        y = self.excitation(y).view(batch_size, channel, 1, 1)
        return x * y

class BandGroupConv(nn.Module):
    def __init__(self, in_channels, groups, groups_channels, num_layers):
        super(BandGroupConv, self).__init__()
        mid_channels = groups_channels * groups
        module_list = nn.ModuleList([])
        module_list.append(SimpleGroupConv(in_channels, mid_channels, groups))
        for _ in range(num_layers - 1):
           module_list.append(SimpleGroupConv(mid_channels, mid_channels, groups))

        module_list.append(SimpleGroupConv(mid_channels, in_channels//3, groups))

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

class SimpleGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(SimpleGroupConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
