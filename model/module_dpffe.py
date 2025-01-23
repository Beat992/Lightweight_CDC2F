from math import sqrt
import torch
import torch.nn as nn
from model.layers import Transformer, TransformerDecoder

'''
    input: DCT patches which need to refinement, some of which is 0,
           means they are either change completely or no change 
    shape: (bs*16) * patch_n * 3*hf
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
        self.t1_f_ex = None
        self.t0_f_ex = None
        self.t1_f = None
        self.t0_f = None
        self.dct_size = dct_size
        self.block_size = block_size
        self.bands = dct_size ** 2
        self.mini_patch_num = (block_size // dct_size) ** 2
        token_len_p = 3 * self.bands
        self.f_conv_band = nn.Conv2d(self.bands * 3, 256, (3, 3))
        self.band_group_conv0 = BandGroupConv(self.bands * 3, 16, 8, 1)
        self.band_group_conv1 = BandGroupConv(self.bands * 3, 4, 32, 1)
        self.band_group_conv2 = BandGroupConv(self.bands * 3, 1, 128, 1)
        self.embedding_layer_p2p = nn.Linear(token_len_p, token_len_p)
        self.encoder_p = Transformer(dim=token_len_p,
                                     depth=encoder_depth,
                                     heads=encoder_heads,
                                     dim_head=encoder_dim,
                                     mlp_dim=token_len_p,
                                     dropout=dropout)
        self.enc_pos_embedding_p2p = nn.Parameter(torch.randn(1, (block_size // dct_size) ** 2, token_len_p))
        self.dec_pos_embedding_p2p = nn.Parameter(torch.randn(1, (block_size // dct_size) ** 2, token_len_p))
        self.decoder_p = TransformerDecoder(dim=token_len_p, depth=decoder_depth, heads=decoder_heads,
                                            dim_head=decoder_dim, dropout=dropout, mlp_dim=decoder_dim)
        self.dc_conv_p = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.st_embedding = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.p_embedding = nn.Sequential(
            nn.Linear(self.bands, 64),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        self.st2fil = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm2d((block_size//dct_size)**2),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.seblock = SEBlock(channel=self.bands*3*2, reduction=6)
        self.conv_t0 = TwoLayerGroupConv(block_size, dct_size)
        self.conv_t1 = TwoLayerGroupConv(block_size, dct_size)
        self.conv_attn = TwoLayerGroupConv(block_size, dct_size)
        self.conv_tov_t0 = TwoLayerGroupConv(block_size, dct_size)
        self.conv_tov_t1 = TwoLayerGroupConv(block_size, dct_size)
        self.cat_attn = TwoLayerGroupConv(block_size, dct_size, True)
        
    def _band_level(self, x):
        bs_block, patch_num, ch, fre_num = x.shape
        h = self.block_size // self.dct_size
        x = x.transpose(1, 3).reshape(bs_block, -1, h, h)
        x = self.band_group_conv0(x) + self.band_group_conv1(x) + self.band_group_conv2(x) + x
        x = self.seblock(x)
        x = x.reshape(bs_block, -1, ch, patch_num).transpose(1, 3)
        return x

    def _p2p(self, x1):
        bs16, patch_num, ch, fre_num = x1.shape
        x1 = x1.reshape(bs16, patch_num, -1)
        m1 = torch.clone(x1)
        x1 = self.embedding_layer_p2p(x1)
        x1 = x1 + self.enc_pos_embedding_p2p
        x1 = self.encoder_p(x1)
        x1 = self.decoder_p(x1, m1 + self.dec_pos_embedding_p2p)
        x1 = x1.view(bs16, patch_num, ch, -1)
        return x1

    def _st2filter(self, x):
        bs16, patch_num, ch, fre_num = x.shape
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        min_vals = x.min(dim=-1, keepdim=True)[0]
        max_vals = x.max(dim=-1, keepdim=True)[0]
        peak_frequency = torch.argmax(x, dim=-1, keepdim=True)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        energy = torch.sum(torch.square(x), dim=-1, keepdim=True)
        # st = torch.cat([var, mean, max_vals, min_vals], dim=-1)
        st = torch.cat([var, mean, max_vals, min_vals, peak_frequency, rms, energy], dim=-1)
        st = self.st_embedding(st)
        x_emb = self.p_embedding(x)
        fil = self.st2fil(st + x_emb).view(bs16, patch_num, 3, 1)
        x = fil * x
        x = x.reshape(bs16, patch_num, ch, fre_num)
        return x

    def _fre_interaction(self, x1, x2):
        x1 = x1.view(-1, self.mini_patch_num, 6 * self.dct_size ** 2).transpose(1, 2).reshape(-1,
                                                                                              6 * self.dct_size ** 2,
                                                                                              self.block_size // self.dct_size,
                                                                                              self.block_size // self.dct_size)
        x2 = x2.view(-1, self.mini_patch_num, 6 * self.dct_size ** 2).transpose(1, 2).reshape(-1,
                                                                                              6 * self.dct_size ** 2,
                                                                                              self.block_size // self.dct_size,
                                                                                              self.block_size // self.dct_size)
        x1_q = self.conv_t0(x1)
        x2_q = self.conv_t1(x2)
        # attn1 = self.cat_attn(torch.cat([x1_q, x2_q], dim=1))
        attn1 = self.cat_attn(torch.cat([x1_q.unsqueeze(2), x2_q.unsqueeze(2)], dim=2).reshape(-1, 12 * self.dct_size ** 2,
                                                                                               self.block_size // self.dct_size,
                                                                                               self.block_size // self.dct_size))
        attn2 = self.conv_attn(x1_q - x2_q)
        attn = torch.sigmoid(attn1 + attn2)
        self.difference_attention_map = attn
        x1 = self.conv_tov_t0(x1) * (attn + 1)
        x2 = self.conv_tov_t1(x2) * (attn + 1)
        x1 = x1.reshape(-1, 6 * self.dct_size ** 2, self.mini_patch_num).transpose(1, 2).reshape(-1, self.mini_patch_num, 6,
                                                                                 self.dct_size ** 2)
        x2 = x2.reshape(-1, 6 * self.dct_size ** 2, self.mini_patch_num).transpose(1, 2).reshape(-1, self.mini_patch_num, 6,
                                                                                 self.dct_size ** 2)
        return x1, x2

    def forward(self, t0, t1, idx):
        # <------inter patch------>
        t0_f = self._f2f(t0)
        t1_f = self._f2f(t1)

        # <------intra patch------>
        t0_dc_p = t0[:, :, :, 0].transpose(1, 2).\
            reshape(idx.numel(), 3, self.block_size // self.dct_size, self.block_size // self.dct_size)
        t1_dc_p = t1[:, :, :, 0].transpose(1, 2).\
            reshape(idx.numel(), 3, self.block_size // self.dct_size, self.block_size // self.dct_size)
        t0_dc_p = self.dc_conv_p(t0_dc_p)
        t1_dc_p = self.dc_conv_p(t1_dc_p)

        t0 = self._st2filter(t0)
        t1 = self._st2filter(t1)

        t0_hp = self._p2p(t0)
        t1_hp = self._p2p(t1)

        t0_dc_p = t0_dc_p.reshape(idx.numel(), 3, (self.block_size // self.dct_size) ** 2).transpose(1, 2)
        t1_dc_p = t1_dc_p.reshape(idx.numel(), 3, (self.block_size // self.dct_size) ** 2).transpose(1, 2)

        t0_hp[:, :, :, 0] = t0_hp[:, :, :, 0] + t0_dc_p
        t1_hp[:, :, :, 0] = t1_hp[:, :, :, 0] + t1_dc_p

        t0 = torch.cat([t0_f, t0_hp], dim=2)
        t1 = torch.cat([t1_f, t1_hp], dim=2)
        self.t0_f = t0
        self.t1_f = t1
        # <-------双时相频域交互------->
        t0, t1 = self._fre_interaction(t0, t1)
        # self.t0_f_ex = t0
        # self.t1_f_ex = t1
        return torch.cat([t0, t1], dim=1)


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

class TwoLayerGroupConv(nn.Module):
    def __init__(self, patch_size, dct_size, if_cat=False):
        super(TwoLayerGroupConv, self).__init__()
        if if_cat:
            factor = 2
        else:
            factor = 1
        self.conv = nn.Sequential(
            nn.Conv2d(6*self.bands * factor, 6*self.bands * 8, 3, 1, 1, groups=3*self.bands),
            nn.BatchNorm2d(6*self.bands * 8),
            nn.ReLU(),
            nn.Conv2d(6*self.bands * 8, 6*self.bands, 3, 1, 1, groups=3*self.bands)
        )
    def forward(self, x):
        return self.conv(x)

class BandGroupConv(nn.Module):
    def __init__(self, in_channels, groups, groups_channels, num_layers):
        super(BandGroupConv, self).__init__()
        mid_channels = groups_channels * groups
        layers = [SimpleGroupConv(in_channels, mid_channels, groups)]
        for _ in range(num_layers - 1):
           layers.append(SimpleGroupConv(mid_channels, mid_channels, groups))

        layers.append(SimpleGroupConv(mid_channels, in_channels//3, groups))
        self.conv_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_module(x)

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
