from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct
from torchvision import transforms

from model.help_function import zigzag_extraction, inverse_zigzag


class ToDCT(nn.Module):
    """
        input: origin img
        input size: bs, c, h, w
        output: each patch' dct coff
        output size: bs, patch num, patch len
    """
    def __init__(self, dct_size, patch_size=64):
        super(ToDCT, self).__init__()
        self.dct_size = dct_size
        self.patch_size = patch_size
        self.patch_num_large = (256//patch_size) ** 2
        self.patch_num_mini = (patch_size//dct_size) ** 2

    def forward(self, x1, x2, idx):
        x1 = self._blocky_(x1, self.patch_size)
        x2 = self._blocky_(x2, self.patch_size)
        x1 = x1.reshape(-1, 3, self.patch_size, self.patch_size)
        x2 = x2.reshape(-1, 3, self.patch_size, self.patch_size)
        x1 = torch.index_select(x1, dim=0, index=idx)
        x2 = torch.index_select(x2, dim=0, index=idx)
        x1 = self._to_dct(self._blocky_(x1, self.dct_size))
        x2 = self._to_dct(self._blocky_(x2, self.dct_size))

        # x1lf, x1mf, x1hf = self._to_dct2(self._blocky_ds(x1))
        # x2lf, x2mf, x2hf = self._to_dct2(self._blocky_ds(x2))
        # mask = self._blocky_64(mask, ~idx)
        # mask = mask * (~idx).unsqueeze(1)
        # return x1lf, x1mf, x1hf, x2lf, x2mf, x2hf
        return x1, x2

    @staticmethod
    def _blocky_(x, size):
        bs, ch, h, w = x.shape
        x = x.reshape(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(size, size), dilation=1, padding=0, stride=(size, size))
        x = x.view(bs, ch, size, size, -1).permute(0, 4, 1, 2, 3)

        return x.reshape(bs, -1, ch, size, size)




    # def _to_dct(self, x):
    #     bs, p, c, _, _, = x.shape
    #     x = x.reshape(-1, c, self.dct_size, self.dct_size)
    #     x += 0.5
    #     x *= 255
    #     x = dct.to_ycbcr(x)
    #     x -= 128  # DCT requires that pixel value must in [-128, 127]
    #
    #     bp, c, h, _ = x.shape
    #     x = x.reshape(bp//self.patch_num, self.patch_num, c, h, -1).permute(0, 2, 1, 3, 4)
    #     x = dct.block_dct(x)
    #
    #     b, c, p, h, _ = x.shape
    #     x = x.view(-1, p, h, h)
    #     x = zigzag_extraction(x)
    #     x = x.reshape(b, c, p, -1).transpose(1, 2)

        # return x
    @staticmethod
    def _to_dct(x):
        # x = x + 1
        x = x / 2
        x = x + 0.5
        x = x * 255
        x = x - 128  # DCT requires that pixel value must in [-128, 127]
        x = x.transpose(1, 2)
        x = dct.block_dct(x)
        b, c, p, h, _ = x.shape
        x = x.view(-1, p, h, h)
        x = zigzag_extraction(x)
        x = x.reshape(b, c, p, -1).transpose(1, 2)
        return x
    #
    @staticmethod
    def _to_dct2(x):
        x += 0.5
        x *= 255
        x -= 128  # DCT requires that pixel value must in [-128, 127]
        x = x.transpose(1, 2)
        x = dct.block_dct(x)
        b, c, p, h, _ = x.shape

        filter_l = torch.zeros(b, c, p, h, h)
        jiezhi_l = 1
        for i in range(jiezhi_l):
            filter_l[:, :, :, i, 0:jiezhi_l-i] = 1
        lf = x * filter_l


        filter_h = torch.zeros(b, c, p, h, h)
        jiezhi_h = 1
        for i in range(jiezhi_h, 8):
            filter_h[:, :, :, i, 7-(i-jiezhi_h):8] = 1
        hf = x * filter_h

        filter_m = torch.ones(b, c, p, h, h)
        filter_m = filter_m - filter_l - filter_h
        mf = x * filter_m

        lf = lf.reshape(b, c, p, -1)
        mf = mf.reshape(b, c, p, -1)
        hf = hf.reshape(b, c, p, -1)
        # x = x.reshape(b, c, p, -1)
        # x[:, :, :, 0] = x[:, :, :, 0] / 100
        return lf, mf, hf

    @staticmethod
    def _idct_and_feature_fre_size_restore(x):
        b, n, c, ds2 = x.shape
        ds = int(sqrt(ds2))
        # x = x.view(b, n * c, -1)
        # x = inverse_zigzag(x, dct_size=ds)
        x = x.view(b, n, c, ds, ds)
        # x = dct.block_idct(x).reshape(b, n, c, -1)
        x = dct.block_idct(x).reshape(b, n, c, -1)
        x = x.permute(0, 2, 3, 1).reshape(-1, ds2, n)
        x = F.fold(x, kernel_size=(ds, ds), output_size=(64, 64), dilation=1, padding=0, stride=(ds, ds))
        x = x.view(b, c, 64, 64)

        return (x + 128) / 255





if __name__=='__main__':
    from PIL import Image
    import matplotlib.pyplot as plt


    img_path = r'/home/pan2/zwb/LEVIR_CD-256/train'
    img1 = Image.open(img_path+'/A/16.png')
    img2 = Image.open(img_path+'/B/16.png')
    to_tensor = transforms.ToTensor()
    img1 = to_tensor(img1).view(1, 3, 256, 256)
    img2 = to_tensor(img2).view(1, 3, 256, 256)
    idx = fine_idx = torch.arange(0, 16)
    todct = ToDCT(dct_size=2, patch_size=64)
    x1lf, x2lf = todct(img1, img2, idx)   # bcp-1
    # x1lf = x1lf.transpose(1, 2)
    # x2lf = x2lf.transpose(1, 2)
    x1lf[:, :, :, 0:3] = 0
    x2lf[:, :, :, 0:3] = 0
    x1lf = todct._idct_and_feature_fre_size_restore(x1lf)
    x2lf = todct._idct_and_feature_fre_size_restore(x2lf) #bc6464
    x1lf = x1lf.permute(1, 2, 3, 0).reshape(3, 4096, 16)
    x2lf = x2lf.permute(1, 2, 3, 0).reshape(3, 4096, 16)
    x1lf = F.fold(x1lf, (256, 256), (64, 64), stride=(64, 64)).squeeze()
    x2lf = F.fold(x2lf, (256, 256), (64, 64), stride=(64, 64)).squeeze()
    # x1lf[[0,2]] = 0
    # x2lf[0:1] = 0
    to_pilImage = transforms.ToPILImage()
    x1lf = to_pilImage(x1lf)
    x2lf = to_pilImage(x2lf)
    x1lf.save('x1_lf.png')
    x2lf.save('x2_lf.png')
    plt.imshow(x)
    plt.show()

    # x1_dct = x1_dct.transpose(2, 3)
    # x1_dct = x1lf.reshape(-1, 64, 64)   # if transpose (bc)fp, else (bc)pf
    # x2_dct = x2_dct.transpose(2, 3)
    # x2_dct = x2lf.reshape(-1, 64, 64)   #
    # x1_dct_mean = torch.mean(x1_dct, dim=1).reshape(48, 8, 8)
    # x2_dct_mean = torch.mean(x2_dct, dim=1).reshape(48, 8, 8)
    # x1_dct = F.fold(x1_dct, (64, 64), (8, 8), stride=(8, 8))
    # x2_dct = F.fold(x2_dct, (64, 64), (8, 8), stride=(8, 8))
    # x1_dct = x1_dct.reshape(3, 16, -1).transpose(1, 2)
    # x2_dct = x2_dct.reshape(3, 16, -1).transpose(1, 2)
    # x1_dct = F.fold(x1_dct, (256, 256), (64, 64), stride=(64, 64))
    # x2_dct = F.fold(x2_dct, (256, 256), (64, 64), stride=(64, 64))
    # x1_dct = x1_dct_mean.numpy()
    # x2_dct = x2_dct_mean.numpy()
    # x1_dct[:, 0, 0] = 0
    # x2_dct[:, 0, 0] = 0
    # diff = abs(x1_dct - x2_dct)
    # plt.imshow(x1_dct[1], cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(x2_dct[1], cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(diff[1], cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()