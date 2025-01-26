import torch
import torch.nn.functional as F
import torch.nn as nn
from torchjpeg import dct
from model.help_function import inverse_zigzag


class FreFeaturePred(nn.Module):
    def __init__(self, phase, dct_size=4, block_size=32):
        super(FreFeaturePred, self).__init__()
        self.phase = phase
        self.dct_size = dct_size
        self.block_size = block_size
        self.h = block_size // dct_size
        self.bands = dct_size ** 2
        self.patch_num = (block_size // dct_size) ** 2
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  64, 3, 1, 1),
        )

        self.pred_head_f2s = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, padding_mode='replicate'),
        )

        self.fre_embedding = nn.Sequential(
            nn.Linear(6 * self.bands, 64),
            nn.BatchNorm1d(self.patch_num),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(self.patch_num),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.pred_head_fre = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(self.patch_num),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(self.patch_num),
            nn.ReLU(),
            nn.Linear(64, 1),    # TODO: 这里的输出通道数再考虑一下
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1, padding_mode='replicate'),
        )

    def forward(self, x0, x1):
        f0 = self._idct_and_feature_fre_size_restore(x0)  # bs16, 6, 64, 64
        f1 = self._idct_and_feature_fre_size_restore(x1)
        x_fre = torch.cat([f0.clone(), f1.clone()], dim=1)
        if self.phase != 'train':
            return None, x_fre

        f0_f2s = self.conv(f0)
        f1_f2s = self.conv(f1)
        pred_score_f2s = torch.abs(f0_f2s - f1_f2s)
        pred_score_f2s = self.pred_head_f2s(pred_score_f2s)

        bs_patch = x_fre.shape[0]
        t0_fre = f0.view(bs_patch, -1, 6 * self.dct_size ** 2)
        t1_fre = f1.view(bs_patch, -1, 6 * self.dct_size ** 2)
        t0_fre = self.fre_embedding(t0_fre)
        t1_fre = self.fre_embedding(t1_fre)
        pred_score_fre = self.pred_head_fre(torch.abs(t0_fre - t1_fre))
        pred_score_fre = pred_score_fre.view(bs_patch, 1, self.h, self.h)

        pred_score_fre = self.deconv(pred_score_fre)

        pred_score = pred_score_f2s + pred_score_fre

        return pred_score, x_fre

    def _idct_and_feature_fre_size_restore(self, x):
        bs_block, n, c, _ = x.shape
        x = x.reshape(bs_block, n * c, -1)
        x = inverse_zigzag(x, dct_size=self.dct_size)
        x = x.view(bs_block, n, c, self.dct_size, self.dct_size)
        x = dct.block_idct(x).reshape(bs_block, n, c, -1)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.badns, n)
        x = F.fold(x, kernel_size=(self.dct_size, self.dct_size), output_size=(self.block_size, self.block_size),
                   dilation=1, padding=0, stride=(self.dct_size, self.dct_size))
        x = x.view(bs_block, c, -1)
        min_value = torch.min(x, dim=2, keepdim=True)[0]
        max_value = torch.max(x, dim=2, keepdim=True)[0]
        # norm x to [-1, 1]
        x = 2 * (x - min_value) / (max_value - min_value) - 1
        return x.view(bs_block, c, self.block_size, self.block_size)

