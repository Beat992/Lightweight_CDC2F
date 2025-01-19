import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchjpeg.dct import block_dct
from tqdm import tqdm
from dataset import CDDataset
from model.help_function import size_restore, cut, apply_dct
from model.module_dpffe import DualPerFreFeatureExtractor
from model.to_dct import ToDCT
from module_coarse_detection import CoarseDetection
from model.module_sfif import SpaFreInteractionFusion
from model.module_fph import FreFeaturePred




class Network(nn.Module):
    def __init__(self,
                 backbone='resnet18', stages_num=4,
                 threshold=0.5, phase='val', dct_size=4, block_size=64,
                 encoder_depth=1, encoder_heads=8, encoder_dim=8, decoder_depth=1,
                 decoder_heads=4, decoder_dim=8,
                 dropout=0.5):
        super(Network, self).__init__()
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
        self.dct_patch_size = dct_size
        self.block_size = block_size
        self.patch_num_large = (256 // block_size) ** 2
        self.phase = phase
        self.coarse_detection = CoarseDetection(backbone)
        self.to_dct = ToDCT(dct_size=dct_size, patch_size=block_size)
        self.dpffe = DualPerFreFeatureExtractor(dct_size=dct_size, patch_size=block_size,
                                                         encoder_depth=encoder_depth, encoder_heads=encoder_heads, encoder_dim=encoder_dim,
                                                         decoder_depth=decoder_depth, decoder_heads=decoder_heads, decoder_dim=decoder_dim, dropout=dropout)
        self.fph = FreFeaturePred(dct_size=dct_size, patch_size=block_size)
        self.sfif = SpaFreInteractionFusion(patch_size=block_size)
        self.pred_head2 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1,),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
        )
        # self.fine_tune = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=32, kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        #     # nn.BatchNorm2d(2),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, ),
        # )

    def forward(self, x0, x1):
        b = int(x0.shape[0])
        """
        <----------- stage one --> coarse detection ---------->
        """
        f0_spa, f1_spa, coarse_score = self.coarse_detection(x0, x1)
        '''
        <---------- block cut and DCT for every block ----------- >
        '''
        fine_idx2, mask = self.get_filter_idx(coarse_score)
        fine_idx = torch.arange(0, b * self.patch_num_large).cuda()
        if fine_idx2.numel() == 0:
            if self.phase == 'val' or 'test':
                coarse2_score_ = 0
                coarse3_score = 0
                fine = torch.sigmoid(coarse_score)
                return coarse_score, coarse2_score_, coarse3_score, fine, fine_idx2
            else:
                fine_idx2 = torch.arange(0, b*self.patch_num_large).cuda()
        else:
            if self.phase == 'val' or 'test':
                fine_idx = fine_idx2

        '''
        <---------- block cut and DCT ----------- >
        '''
        x = torch.cat([x0, x1], dim=1)    # N, C+C, H, H
        blocks = cut(x, self.block_size)          # N, L, C+C, block_size, block_size
        b0, b1 = self.select_and_dct(blocks, fine_idx).chunk(2, dim=2)
        '''
        <---------- stage two --> fine detection-------->
        '''
        f1, f2 = self.dpffe()
        '''
        <---------- feature from fre transformer pred---------->
        '''
        coarse3_score, x_fre = self.frepred(x_fre)       # bs16, 1, 64, 64
        if self.phase == 'train':
            coarse3_score = coarse3_score.view(b, -1, self.block_size ** 2)
            coarse3_score = size_restore(coarse3_score, input_size=self.block_size, output_size=256)
        else:
            coarse3_score = 0
        # coarse3_prob = torch.sigmoid(coarse3_score)
        '''
        < ---------- fre feature and spa feature interaction---------->
        '''
        # x_fre = torch.index_select(x_fre, 0, fine_idx2)
        t0, t1 = self.sfifm(feature, x_fre, coarse_prob_sel, fine_idx2)
        coarse2_score_ = torch.abs(t0 - t1)
        coarse2_score_ = self.pred_head2(coarse2_score_)
        coarse2_score = torch.zeros(b * self.patch_num_large, 1, self.block_size, self.block_size).to(coarse2_score_.dtype).cuda()
        coarse2_score[fine_idx2, :, :, :] = coarse2_score_
        coarse2_score = coarse2_score.view(b, -1, self.block_size ** 2)
        coarse2_score = size_restore(coarse2_score, input_size=self.block_size, output_size=256)
        '''
        <-------- three branch pred cm fusion------->
        '''
        fine = torch.sigmoid(coarse_score + coarse2_score)

        # return coarse_prob, coarse2_score, coarse3_score, fine, fine_idx2
        return coarse_score, coarse2_score_, coarse3_score, fine, fine_idx2
        # return coarse3_score

    def block_filter(self, coarse_score, fine_idx2):
        coarse_prob_sel = F.unfold(coarse_score, kernel_size=(self.block_size, self.block_size),
                                   padding=0, stride=(self.block_size, self.block_size))
        coarse_prob_sel = coarse_prob_sel.transpose(1, 2).reshape(-1, self.block_size, self.block_size)
        coarse_prob_sel = torch.index_select(coarse_prob_sel, dim=0, index=fine_idx2.squeeze())
        return coarse_prob_sel


    def get_filter_idx(self, coarse_score):
        coarse_prob = torch.sigmoid(coarse_score)
        coarse_mask = torch.tensor(coarse_prob > self.threshold)
        fine_index = None
        if self.phase == 'train':  # 训练时用所有patch
            fine_index = torch.ones(coarse_mask.shape[0] * (256 // self.block_size) ** 2).cuda()
        elif self.phase == 'val' or 'test':  # 推理时进行筛选
            fine_index = F.unfold(coarse_mask, kernel_size=(self.block_size, self.block_size),
                                  padding=0, stride=(self.block_size, self.block_size))  # bs * 4096 * 16
            fine_index = (fine_index.transpose(1, 2).sum(dim=2)).flatten()
            fine_index = torch.gt(fine_index, 0) & torch.lt(fine_index, self.block_size ** 2)

        mask = torch.ones(coarse_mask.shape[0] * (256 // self.block_size) ** 2, 1, self.block_size,
                          self.block_size).cuda()
        mask = (mask * fine_index.view(coarse_mask.shape[0] * (256 // self.block_size) ** 2, 1, 1, 1))
        fine_index = torch.nonzero(fine_index)

        mask = mask.view(coarse_mask.shape[0], (256 // self.block_size) ** 2, -1).transpose(1, 2)
        mask = F.fold(mask, kernel_size=(self.block_size, self.block_size), output_size=(256, 256),
                      dilation=1, padding=0,
                      stride=(self.block_size, self.block_size))

        # return coarse_prob, coarse_score, fine_index, finetune_mask, spafeature, raw_feature
        # coarse_score_selection = coarse_score_selection.view(-1, self.patch_num, self.patch_size, self.patch_size)
        return fine_index.squeeze(), mask

    def select_and_dct(self, x, fine_idx):
        """
        根据idx筛选block, 并且做dct
        :param x: 切分后的block, shape: N * block_num(B) * C * block_size * block_size
        :param fine_idx: 保留的block的索引
        :return: x的dct并且做zigzag展开成一维向量, shape: (NB) * patch_num * C * dct_size^2
        """
        b, l, c, block_size, block_size = x.shape
        x = torch.index_select(x, dim=0, index=fine_idx)
        x = x.view(-1, c, block_size, block_size)
        x = apply_dct(x, self.dct_patch_size)
        return x






if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile

    A = torch.randn(1, 3, 256, 256).cuda()
    B = torch.randn(1, 3, 256, 256).cuda()
    model = Network('resnet18', 4, 0.5, 'train', 4, 32, 4, 8, 16, 1, 8, 16, 0.5).cuda()
    summary(model=model, input_size=[(3, 256, 256), (3, 25, 256)], batch_size=1, device="cuda")
    flops, params = profile(model, inputs=(A, B))
    print(f"模型的FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位显示
    print(f"模型的参数数量: {params / 1e6} M")
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))