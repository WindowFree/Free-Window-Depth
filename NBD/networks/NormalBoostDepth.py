import pytorch_lightning as pl

import torch
import torch.nn as nn
import numpy as np

import timm
from .heads import *
from .swin_transformer import SwinTransformer
# from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .blocks import NormalImageEmbedBlock
from .QCRF_layers import QCRF, JoinQCRF, JoinQCRF2
from .gru import DepthRefineBlock
from .newcrf_layers import NewCRF
                
class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale=1):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

# -------ours-------
class NormalBoostDepth_v5_QCRF_GRU(nn.Module):
    def __init__(self, input_height = 352, input_width=1120, version = 'large', window_size=7, min_depth=0.1, max_depth = 100.,  gru_epochs=2, pretrained=None, frozen_stages = -1, iters=4):
        super().__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        if version == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        
        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        
        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )
        
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = QCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = QCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = QCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = QCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.crf7 = QCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf6 = QCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf5 = QCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf4 = QCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.psp = PSP(**decoder_cfg)
        self.disp_head = DispHead(input_dim=crf_dims[0])
        self.up_mode = 'bilinear'

        self.normal_head = nn.Sequential(
            nn.Conv2d(crf_dims[0], crf_dims[0]//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(crf_dims[0]//2, 3, 3, 1, 1)
        )

        self.nie = NormalImageEmbedBlock(features=crf_dims[0], input_h=input_height, input_w=input_width, kernel_size_1=8, kernel_size_2=1, num_blocks=8)
        self.pro_proj = nn.Conv2d(crf_dims[0], crf_dims[0], 1, 1)
        self.min_depth = self.min_depth
        self.max_depth = self.max_depth
        self.depth_refine = DepthRefineBlock(itrs=iters)
        self.gru_epochs = gru_epochs
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, imgs, epoch):
        feats = self.backbone(imgs)
        
        ppm = self.psp(feats)
        # normal feature
        e3 = self.crf3(feats[3], ppm)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1)
        # depth feature
        e7 = self.crf7(feats[3], ppm)
        e7 = nn.PixelShuffle(2)(e7)
        e6 = self.crf6(feats[2], e7)
        e6 = nn.PixelShuffle(2)(e6)
        e5 = self.crf5(feats[1], e6)
        e5 = nn.PixelShuffle(2)(e5)
        e4 = self.crf4(feats[0], e5)

        normal = self.normal_head(e0)
        depth_feature = self.nie(normal, e4, 1)
        depth = self.disp_head(depth_feature) 

        normal = F.interpolate(normal, scale_factor=4, mode='bilinear', align_corners=True)
        
        if epoch < self.gru_epochs:
            depth = depth * (self.max_depth - self.min_depth) + self.min_depth
            output = {
                'depth': depth,
                'normal': normal,
            }
        else:
            depth_list = self.depth_refine(depth, torch.cat([e0, e4], dim=1))
            for idx, depth in enumerate(depth_list):
                depth_list[idx] = depth_list[idx] * (self.max_depth - self.min_depth) + self.min_depth
            output = {
                'depth': depth_list,
                'normal': normal
            }
        return output

    def get_parameters(self):
        return self.parameters()
    
    def get_backbone_parameters(self):
        yield from self.backbone.parameters()

    def get_other_parameters(self):
        backbone_params = set(self.get_backbone_parameters())
        for param in self.parameters():
            if param not in backbone_params:
                yield param
     
