import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class NormalImageCrossAttentionBlock(nn.Module):
    # 交叉注意力
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.cross_query_linear = nn.Linear(embed_size, embed_size)
        self.cross_key_linear = nn.Linear(embed_size, embed_size)
        self.cross_value_linear = nn.Linear(embed_size, embed_size)
        self.heads = 8
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        
        self.LayerNorm = nn.LayerNorm(embed_size)

        self.init_weights()

    def forward(self, image_feature, normal_feature):
        Q = self.cross_query_linear(image_feature)
        K = self.cross_key_linear(normal_feature)
        V = self.cross_value_linear(normal_feature)
        
        cross_att_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.embed_size)
        cross_att_score = torch.softmax(cross_att_score, dim=-1)
        cross_feature = torch.matmul(cross_att_score, V)
        
        output = self.fc(cross_feature) + image_feature
        output = self.LayerNorm(output)
        return output
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.cross_query_linear.weight)
        nn.init.xavier_uniform_(self.cross_key_linear.weight)
        nn.init.xavier_uniform_(self.cross_value_linear.weight)
        nn.init.zeros_(self.cross_query_linear.bias)
        nn.init.zeros_(self.cross_key_linear.bias)
        nn.init.zeros_(self.cross_value_linear.bias)


class NormalImageEmbedBlock(nn.Module):
    def __init__(self, features, input_h, input_w, kernel_size_1, kernel_size_2, num_blocks):
        super().__init__()
        assert kernel_size_1 % 2 == 0
        self.normal_embed_conv = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=kernel_size_1, stride=kernel_size_1),
            nn.GELU(),
        )
        self.feature_embed_conv = nn.Conv2d(features, features, kernel_size=kernel_size_2, stride=kernel_size_2)
        self.pos_embed = nn.Parameter(torch.randn(1, input_h // 32 * input_w // 32, features).contiguous())

        self.blocks = nn.ModuleList(
            NormalImageCrossAttentionBlock(features) for _ in range(num_blocks)
        )
        
        self.layer_norm = nn.LayerNorm(features)
        
    def forward(self, normal, feature, up_scale = 1):
        normal = normal.detach()
        normal_embed = self.normal_embed_conv(normal) # B features H/32 W/32

        normal_embed = rearrange(normal_embed, 'b c h w -> b (h w) c').contiguous() # B (H * W / 1024) features
        normal_embed = self.layer_norm(normal_embed)
        normal_embed = normal_embed + self.pos_embed
        feature_embed = self.feature_embed_conv(feature) # B features H/4 W/4
        h, w = feature_embed.size(-2), feature_embed.size(-1)
        feature_embed = rearrange(feature_embed, 'b c h w -> b (h w) c').contiguous()
        feature_embed = self.layer_norm(feature_embed)
        for block in self.blocks:
            output = block(feature_embed, normal_embed)
            feature_embed = output
        
        feature_embed = rearrange(feature_embed, 'b (h w) c -> b c h w', h=h).contiguous()
        if up_scale != 1:
            feature_embed = F.interpolate(feature_embed, scale_factor=up_scale, mode='bilinear', align_corners=True)
        
        return feature_embed

 