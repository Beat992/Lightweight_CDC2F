import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

class CDC2F(nn.Module):
    def __init__(self,
                 backbone='resnet18', stages_num=4, backbone_pretrained=False,
                 threshold=0.5, phase='val', dct_size=4, block_size=32,
                 encoder_depth=1, encoder_heads=8, encoder_dim=8, decoder_depth=1,
                 decoder_heads=4, decoder_dim=8,
                 dropout=0.5):
        super(CDC2F, self).__init__()
        self.batch_size = None
        if stages_num == 4:
            if backbone == 'resnet18':
                self.channel = 512
            else:
                self.channel = 1856
        elif stages_num == 5:
            if backbone == 'resnet18':
                self.channel = 1024
            else:
                self.channel = 3904
        self.threshold = threshold
        self.dct_patch_size = dct_size
        self.block_size = block_size
        self.block_num = (256 // block_size) ** 2
        self.bands = dct_size ** 2
        self.phase = phase
        self.coarse_detection = CoarseDetection(backbone, backbone_pretrained)
        self.dpffe = DualPerFreFeatureExtractor(dct_size, block_size,
                                                encoder_depth, encoder_heads, encoder_dim,
                                                decoder_depth, decoder_heads, decoder_dim, dropout=dropout)
        self.fie = FreInfoExchange(block_size, dct_size)
        self.fph = FreFeaturePred(phase, dct_size, block_size)
        self.sfif = SpaFreInteractionFusion(block_size)
        self.conv_fused_feature = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        self.pred_head_fine = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x0, x1):
        self.batch_size = x0.shape[0]
        """
        <----------- stage one --> coarse detection ---------->
        """
        f0_spa, f1_spa, coarse_score = self.coarse_detection(x0, x1)
        '''
        <---------- block cut and DCT for every block ----------- >
        '''
        fine_idx, mask = self.get_filter_idx(coarse_score)
        coarse_prob_retaining = self.get_prob(coarse_score, fine_idx)
        if fine_idx.numel() == 0:     # only possible happen in val or test, so return coarse score as fine
            return coarse_score
        '''
        <---------- block cut and DCT ----------- >
        '''
        x = torch.cat([x0, x1], dim=1)    # N, C+C, H, H
        blocks = cut(x, self.block_size)          # N, L, C+C, block_size, block_size
        b0, b1 = self.filter_and_dct(blocks, fine_idx).chunk(2, dim=2)
        '''
        <---------- stage two --> fine detection-------->
        '''
        f0_fre, f1_fre = self.dpffe(b0, b1)
        f0_fre, f1_fre = self.fie(f0_fre, f1_fre)
        '''
        <---------- extra task --> frequency domain pred---------->
        '''
        fre_score, f_fre = self.fph(f0_fre, f1_fre)       # bs_block, 1, self.block_size, self.block_size
        if fre_score is not None:
            fre_score = fre_score.view(self.batch_size, self.block_num, -1)
            fre_score = size_restore(fre_score, input_size=self.block_size, output_size=256)
        '''
        < ---------- dual domain feature interaction fusion---------->
        '''
        f0_spa, f1_spa = self.filter_spa_feature(f0_spa, f1_spa, fine_idx)
        f0_fre, f1_fre = f_fre.chunk(2, dim=1)  # bs_block, c, self.block_size, self.block_size
        f0 = self.sfif(f0_spa, f0_fre, coarse_prob_retaining)
        f1 = self.sfif(f1_spa, f1_fre, coarse_prob_retaining)

        f0, f1 = self.conv_fused_feature(f0), self.conv_fused_feature(f1)
        fine_score_filtered = torch.abs(f0 - f1)
        fine_score_filtered = self.pred_head_fine(fine_score_filtered)
        fine_score = torch.zeros(self.batch_size * self.block_num, 1, self.block_size, self.block_size).to(fine_score_filtered.dtype).cuda()

        if self.phase == 'train':
            fine_score_filtered = fine_score_filtered.view(self.batch_size, self.block_num, -1)
            fine_score_filtered = size_restore(fine_score_filtered, self.block_size, 256)
            return coarse_score, fre_score, fine_score_filtered, fine_idx
        else:
            fine_score[fine_idx, :, :, :] = fine_score_filtered
            fine_score = size_restore(fine_score.view(self.batch_size, -1, self.block_size**2), self.block_size, 256)

        return torch.sigmoid(coarse_score + fine_score)


    def filter_spa_feature(self, f0_spa, f1_spa, idx):
        f_spa = torch.cat([f0_spa, f1_spa], dim=1)
        ch = f_spa.shape[1]
        f_spa = F.unfold(f_spa, kernel_size=(self.block_size // 2, self.block_size // 2), padding=0,
                         stride=(self.block_size // 2, self.block_size // 2))
        f_spa = f_spa.transpose(1, 2).reshape(-1, ch, self.block_size // 2, self.block_size // 2)
        f_spa = torch.index_select(f_spa, 0, idx)
        return f_spa.chunk(2, dim=1)

    def get_prob(self, coarse_score, fine_idx):
        score = F.unfold(coarse_score, kernel_size=(self.block_size, self.block_size),
                                   padding=0, stride=(self.block_size, self.block_size))
        score = score.transpose(1, 2).reshape(-1, self.block_size, self.block_size)
        score = torch.index_select(score, dim=0, index=fine_idx)
        return torch.sigmoid(score)

    def get_filter_idx(self, coarse_score):
        coarse_prob = torch.sigmoid(coarse_score)
        coarse_mask = (coarse_prob > self.threshold).float()
        # ps: only filtering when validation or testing, using all block when training
        idx = torch.ones(self.batch_size * self.block_num).cuda()
        if self.phase == 'val' or self.phase == 'test':
            idx = F.unfold(coarse_mask, kernel_size=(self.block_size, self.block_size),
                                  padding=0, stride=(self.block_size, self.block_size))
            idx = (idx.transpose(1, 2).sum(dim=2)).flatten()
            idx = torch.gt(idx, 0) & torch.lt(idx, self.block_size ** 2)

        mask = torch.ones(self.batch_size * self.block_num, self.block_size, self.block_size).cuda()
        mask = (mask * idx.view(self.batch_size * self.block_num, 1, 1))
        mask = mask.view(self.batch_size, self.block_num, -1).transpose(1, 2)
        mask = F.fold(mask, kernel_size=(self.block_size, self.block_size), output_size=(256, 256),
                      dilation=1, padding=0,
                      stride=(self.block_size, self.block_size))
        idx = torch.nonzero(idx)

        return idx.squeeze(), mask

    def filter_and_dct(self, x, fine_idx):
        """
        根据idx筛选block, 并且做dct
        :param x: 切分后的block, shape: N * block_num(B) * C * block_size * block_size
        :param fine_idx: 保留的block的索引
        :return: x的dct并且做zigzag展开成一维向量, shape: (NB) * patch_num * C * dct_size^2
        """
        b, l, c, block_size, block_size = x.shape
        x = x.reshape(-1, c, block_size, block_size)
        x = torch.index_select(x, dim=0, index=fine_idx)
        x = apply_dct(x, self.dct_patch_size)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile
    import configs as cfg

    A = torch.randn(1, 3, 256, 256).cuda()
    B = torch.randn(1, 3, 256, 256).cuda()
    model = CDC2F(cfg.backbone, 5, False)
    summary(model=model, input_size=[(3, 256, 256), (3, 25, 256)], batch_size=1, device='cuda')
    flops, params = profile(model, inputs=(A, B))
    print(f"模型的FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位显示
    print(f"模型的参数数量: {params / 1e6} M")
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))