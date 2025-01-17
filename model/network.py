import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset.LEVIR_CD import CDDataset
from model.help_function import size_restore
from model.network_coarse import Coarse
from model.Dual_Perspective_Frequency_Feature_Extractor import FreTransformer
# from model.cdd_Dpffe import FreTransformer
from model.to_dct import ToDCT
from model.backbone import ResNet
from model.Spa_Fre_Feature_Fusion import SpaFreInteractionFusionModule
# from model.cdd_Sfif import SpaFreInteractionFusionModule
from configs.LEVIR_CD import pretrain_model_path_ResNet18
from model.Fre_Feature_Pred_Head import FreFeaturePred
# from model.cdd_fre_pred import FreFeaturePred


class Network(nn.Module):
    def __init__(self,
                 backbone='resnet18', stages_num=4,
                 threshold=0.5, phase='val', dct_size=4, patch_size=64,
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
        self.dct_size = dct_size
        self.patch_size = patch_size
        self.patch_num_large = (256 // patch_size) ** 2
        self.phase = phase
        self.backbone = ResNet(backbone=backbone, resnet_stages_num=stages_num)
        self.coarse = Coarse(stages_num=stages_num, threshold=threshold, patch_size=patch_size, channel=self.channel,
                             phase=phase)
        self.to_dct = ToDCT(dct_size=dct_size, patch_size=patch_size)
        self.freTransformer = FreTransformer(dct_size=dct_size, patch_size=patch_size,
                                             encoder_depth=encoder_depth, encoder_heads=encoder_heads, encoder_dim=encoder_dim,
                                             decoder_depth=decoder_depth, decoder_heads=decoder_heads, decoder_dim=decoder_dim, dropout=dropout)
        self.frepred = FreFeaturePred(dct_size=dct_size, patch_size=patch_size)
        self.sfifm = SpaFreInteractionFusionModule(patch_size=patch_size)
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
        pretrain_dict = torch.load(pretrain_model_path_ResNet18)
        self.backbone.resnet.load_state_dict(pretrain_dict)

    def forward(self, x1, x2):
        b = int(x1.shape[0])

        """
        <----------- backbone and pred head 1 ---------->
        """
        feature = [self.backbone(x1), self.backbone(x2)]
        coarse_prob, coarse_score, fine_idx2, feature, finetune_mask = self.coarse(feature)
        fine_idx = torch.arange(0, b * self.patch_num_large).cuda()
        '''
        <---------- blocky 64 and DCT for every 64x patch ----------- >
        '''
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
        coarse_prob_sel = F.unfold(coarse_score, kernel_size=(self.patch_size, self.patch_size),
                                          padding=0, stride=(self.patch_size, self.patch_size))
        coarse_prob_sel = coarse_prob_sel.transpose(1, 2).reshape(-1, self.patch_size, self.patch_size)
        coarse_prob_sel = torch.index_select(coarse_prob_sel, dim=0, index=fine_idx2.squeeze())

        x1, x2 = self.to_dct(x1, x2, fine_idx)
        '''
        <----------two kind of transformer block-------->
        '''
        x_fre = self.freTransformer(x1, x2, fine_idx)
        '''
        <---------- feature from fre transformer pred---------->
        '''
        coarse3_score, x_fre = self.frepred(x_fre)       # bs16, 1, 64, 64
        if self.phase == 'train':
            coarse3_score = coarse3_score.view(b, -1, self.patch_size**2)
            coarse3_score = size_restore(coarse3_score, input_size=self.patch_size, output_size=256)
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
        coarse2_score = torch.zeros(b*self.patch_num_large, 1, self.patch_size, self.patch_size).to(coarse2_score_.dtype).cuda()
        coarse2_score[fine_idx2, :, :, :] = coarse2_score_
        coarse2_score = coarse2_score.view(b, -1, self.patch_size**2)
        coarse2_score = size_restore(coarse2_score, input_size=self.patch_size, output_size=256)
        '''
        <-------- three branch pred cm fusion------->
        '''
        fine = torch.sigmoid(coarse_score + coarse2_score)

        # return coarse_prob, coarse2_score, coarse3_score, fine, fine_idx2
        return coarse_score, coarse2_score_, coarse3_score, fine, fine_idx2
        # return coarse3_score

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