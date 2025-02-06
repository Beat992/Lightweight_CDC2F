import torch
import torch.nn as nn
from model import *

class PureFreCDNet(nn.Module):
    def __init__(self,
                 threshold=0.5, phase='val', dct_size=4, block_size=32,
                 encoder_depth=1, encoder_heads=8, encoder_dim=8, decoder_depth=1,
                 decoder_heads=4, decoder_dim=8,
                 dropout=0.5):
        super(PureFreCDNet, self).__init__()
        self.batch_size = None
        self.threshold = threshold
        self.dct_patch_size = dct_size
        self.block_size = block_size
        self.block_num = (256 // block_size) ** 2
        self.bands = dct_size ** 2
        self.phase = phase
        self.dpffe = DualPerFreFeatureExtractor(dct_size, block_size,
                                                encoder_depth, encoder_heads, encoder_dim,
                                                decoder_depth, decoder_heads, decoder_dim, dropout=dropout)
        self.fie = FreInfoExchange(block_size, dct_size)
        self.fph = FreFeaturePred(phase, dct_size, block_size)

    def forward(self, x0, x1):
        self.batch_size = x0.shape[0]
        '''
        <---------- block cut and DCT ----------- >
        '''
        x = torch.cat([x0, x1], dim=1)    # N, C+C, H, H
        blocks = cut(x, self.block_size)          # N, L, C+C, block_size, block_size
        b0, b1 = self.dct(blocks).chunk(2, dim=2)
        '''
        <---------- stage two --> fine detection-------->
        '''
        f0_fre, f1_fre = self.dpffe(b0, b1)
        f0_fre, f1_fre = self.fie(f0_fre, f1_fre)
        '''
        <---------- extra task --> frequency domain pred---------->
        '''
        fre_score, f_fre = self.fph(f0_fre, f1_fre)       # bs_block, 1, self.block_size, self.block_size
        fre_score = fre_score.view(self.batch_size, self.block_num, -1)
        fre_score = size_restore(fre_score, input_size=self.block_size, output_size=256)


        return torch.sigmoid(fre_score)

    def dct(self, x):
        """
        根据idx筛选block, 并且做dct
        :param x: 切分后的block, shape: N * block_num(B) * C * block_size * block_size
        :param fine_idx: 保留的block的索引
        :return: x的dct并且做zigzag展开成一维向量, shape: (NB) * patch_num * C * dct_size^2
        """
        b, l, c, block_size, block_size = x.shape
        x = x.reshape(-1, c, block_size, block_size)
        x = apply_dct(x, self.dct_patch_size)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile
    import configs as cfg

    A = torch.randn(1, 3, 256, 256).cuda()
    B = torch.randn(1, 3, 256, 256).cuda()
    model = PureFreCDNet(cfg.backbone, 5, False)
    summary(model=model, input_size=[(3, 256, 256), (3, 25, 256)], batch_size=1, device='cuda')
    flops, params = profile(model, inputs=(A, B))
    print(f"模型的FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位显示
    print(f"模型的参数数量: {params / 1e6} M")
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))