import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from .newcrf_layers import NewCRF


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RectifyCoordsGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, coords_lambda=20):
        ctx.in1 = coords_lambda
        ctx.save_for_backward(coords)
        return coords

    @staticmethod
    def backward(ctx, grad_output):
        coords_lambda = ctx.in1
        coords, = ctx.saved_tensors
        grad_output[coords < -1.001] += -coords_lambda * 10
        grad_output[coords > 1.001] += coords_lambda * 10
        return grad_output, None


def calc_rel_pos_spatial(attn, q, q_shape, k_shape, rel_pos_h, rel_pos_w, overlap=0):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    k_h = k_h + 2 * overlap
    k_w = k_w + 2 * overlap

    # Scale up rel pos if shapes for q and k are different.
    # q_h_ratio = max(k_h / q_h, 1.0)
    # k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] - torch.arange(k_h)[None, :]
    )
    dist_h += (k_h - 1)
    # q_w_ratio = max(k_w / q_w, 1.0)
    # k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] - torch.arange(k_w)[None, :]
    )
    dist_w += (k_w - 1)

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w).contiguous()
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w).contiguous()

    return attn


class QuadrangleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_bias=False, attn_drop=0., window_size=7, rpe='v2', coords_lambda=0.01):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.coords_lambda = coords_lambda

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.identity = nn.Identity()  # for hook
        self.identity_attn = nn.Identity()  # for hook
        self.identity_distance = nn.Identity()

        self.transform = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads*9, kernel_size=1, stride=1)
            )

        self.rpe = rpe
        if rpe == 'v1':
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size * 2 - 1) * (window_size * 2 - 1), num_heads))  # (2*Wh-1 * 2*Ww-1 + 1, nH) 
            # self.relative_position_bias = torch.zeros(1, num_heads) # the extra is for the token outside windows

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The v1 relative_pos_embedding is used')
        elif rpe == 'v2':
            q_size = window_size
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=.02)
            trunc_normal_(self.rel_pos_w, std=.02)
            print('The v2 relative_pos_embedding is used')

    def forward(self, x, v_, h, w):
        b, N, C = x.shape # h * w = N
        x = x.reshape(b, h, w, C).permute(0, 3, 1, 2).contiguous() # b c h w
        v_ = v_.reshape(b, h, w, C).permute(0, 3, 1, 2).contiguous() # b c h w
        shortcut = x
        qk_shortcut = F.conv2d(shortcut, self.qk.weight.unsqueeze(-1).unsqueeze(-1), bias=self.qk.bias, stride=1)
        ws = self.window_size
        padding_t = 0
        padding_d = (ws - h % ws) % ws
        padding_l = 0
        padding_r = (ws - w % ws) % ws
        expand_h, expand_w = h+padding_t+padding_d, w+padding_l+padding_r
        window_num_h = expand_h // ws
        window_num_w = expand_w // ws
        assert expand_h % ws == 0
        assert expand_w % ws == 0
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).contiguous().unsqueeze(0)
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, ws, window_num_w, ws)
        window_center_coords = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(ws).to(x.device) * 2 / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(ws).to(x.device) * 2 / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).contiguous().reshape(1, 2, window_num_h, ws, window_num_w, ws).permute(0, 2, 4, 1, 3, 5).contiguous()

        qk = qk_shortcut
        qk = torch.nn.functional.pad(qk, (padding_l, padding_r, padding_t, padding_d))
        v = torch.nn.functional.pad(v_, (padding_l, padding_r, padding_t, padding_d))
        qk = rearrange(qk, 'b (num h dim) hh ww -> num (b h) dim hh ww', h=self.num_heads, num=2, dim=self.dim//self.num_heads, b=b, hh=expand_h, ww=expand_w).contiguous()
        v = rearrange(v, 'b (h dim) hh ww -> (b h) dim hh ww', h=self.num_heads, dim=self.dim//self.num_heads, b=b, hh=expand_h, ww=expand_w).contiguous()
        q, k = qk.unbind(0)

        if h > ws or w > ws:
            x = torch.nn.functional.pad(shortcut, (padding_l, padding_r, padding_t, padding_d))
            sampling_ = self.transform(x).reshape(b*self.num_heads, 9, window_num_h, window_num_w).permute(0, 2, 3, 1).contiguous()
            sampling_offsets = sampling_[..., :2,]
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / (expand_w // ws)
            sampling_offsets[..., 1] = sampling_offsets[..., 1] / (expand_h // ws)

            sampling_offsets = sampling_offsets.reshape(-1, window_num_h, window_num_w, 2, 1).contiguous()
            sampling_scales = sampling_[..., 2:4] + 1
            sampling_shear = sampling_[..., 4:6]
            sampling_projc = sampling_[..., 6:8]
            sampling_rotation = sampling_[..., -1]
            zero_vector = torch.zeros(b*self.num_heads, window_num_h, window_num_w).cuda()
            sampling_projc = torch.cat([
                sampling_projc.reshape(-1, window_num_h, window_num_w, 1, 2),
                torch.ones_like(zero_vector).cuda().reshape(-1, window_num_h, window_num_w, 1, 1)
                ], dim=-1)

            shear_matrix = torch.stack([
                torch.ones_like(zero_vector).cuda(),
                sampling_shear[..., 0],
                sampling_shear[..., 1],
                torch.ones_like(zero_vector).cuda()], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            scales_matrix = torch.stack([
                sampling_scales[..., 0],
                torch.zeros_like(zero_vector).cuda(),
                torch.zeros_like(zero_vector).cuda(),
                sampling_scales[..., 1],
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            rotation_matrix = torch.stack([
                sampling_rotation.cos(),
                sampling_rotation.sin(),
                -sampling_rotation.sin(),
                sampling_rotation.cos()
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            basic_transform_matrix = rotation_matrix @ shear_matrix @ scales_matrix
            affine_matrix = torch.cat(
                (torch.cat((basic_transform_matrix, sampling_offsets), dim=-1), sampling_projc), dim=-2)
            window_coords_pers = torch.cat([
                window_coords.flatten(-2, -1), torch.ones(1, window_num_h, window_num_w, 1, ws*ws).cuda()
            ], dim=-2)
            transform_window_coords = affine_matrix @ window_coords_pers

            _transform_window_coords3 = transform_window_coords[..., -1, :]
            _transform_window_coords3[_transform_window_coords3==0] = 1e-6
            transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)

            transform_window_coords_distance = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws*ws, 1)
            transform_window_coords_distance = transform_window_coords_distance - window_coords.reshape(-1, window_num_h, window_num_w, 2, 1, ws*ws)
            transform_window_coords_distance = torch.sqrt((transform_window_coords_distance[..., 0, :, :]*(expand_w-1)/2) ** 2 + (transform_window_coords_distance[..., 1, :, :]*(expand_h-1)/2) ** 2)
            transform_window_coords_distance = rearrange(transform_window_coords_distance, '(b h) hh ww n1 n2 -> (b hh ww) h n1 n2', b=b, h=self.num_heads, hh=window_num_h, ww=window_num_w, n1=ws*ws, n2=ws*ws).contiguous()
            transform_window_coords = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
            #TODO: adjust the order of transformation

            coords = window_center_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1) + transform_window_coords

            sample_coords = coords.permute(0, 2, 3, 4, 5, 1).contiguous().reshape(b*self.num_heads, ws*window_num_h, ws*window_num_w, 2).contiguous()
            # sample_coords = RectifyCoordsGradient.apply(sample_coords, self.coords_lambda)

            k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
            v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()
            k = rearrange(k_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()
            v = rearrange(v_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()
        else:
            transform_window_coords_distance = None
            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()
            k = rearrange(k, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()
            v = rearrange(v, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws).contiguous()

        attn = (q * self.scale) @ k.transpose(-2, -1).contiguous()
        if self.rpe == 'v1':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).contiguous()].view(self.window_size * self.window_size, self.window_size * self.window_size, -1).contiguous()  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn += relative_position_bias.unsqueeze(0)
            pass
        elif self.rpe == 'v2':
            attn = calc_rel_pos_spatial(attn.float(), q.float(), (self.window_size, self.window_size), (self.window_size, self.window_size), self.rel_pos_h.float(), self.rel_pos_w.float())
        attn = attn.softmax(dim=-1)
        if transform_window_coords_distance is not None:
            transform_window_coords_distance = (transform_window_coords_distance * attn).sum(dim=-1)
            transform_window_coords_distance = self.identity_distance(transform_window_coords_distance)

        out = attn @ v
        out = rearrange(out, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws).contiguous()
        if padding_t + padding_d + padding_l + padding_r > 0:
            out = out[:, :, padding_t:h+padding_t, padding_l:w+padding_l]
        out = out.reshape(b, self.dim, -1).permute(0, 2, 1).contiguous()
        out = self.proj(out)
        return out

    def _reset_parameters(self):
        nn.init.constant_(self.transform[-1].weight, 0.)
        nn.init.constant_(self.transform[-1].bias, 0.)


class QCRFBlock(nn.Module):
    def __init__(self, dim, num_heads, v_dim, window_size=7, mlp_ratio=4., qk_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpe='v1'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.norm1 = norm_layer(dim)
        self.attn = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = QuadrangleAttention(dim, num_heads, qk_bias, attn_drop, window_size, rpe=rpe)

        self.H = None
        self.W = None

    def forward(self, x, v):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)

        attn = self.attn(x, v, H, W)
        x = shortcut + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicQCRFLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, v_dim, window_size=7, mlp_ratio=4., qk_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, rpe='v1'):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            QCRFBlock(
                dim=dim, num_heads=num_heads, v_dim=v_dim, window_size=window_size, mlp_ratio=mlp_ratio, qk_bias=qk_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, rpe=rpe
            ) for _ in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, v, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, v)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class QCRF(nn.Module):
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 v_dim=64,
                 window_size=7,
                 num_heads=4,
                 depth=2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 rpe='v2'):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        if input_dim != embed_dim:
            self.proj_x = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
        else:
            self.proj_x = None

        if v_dim != embed_dim:
            self.proj_v = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        elif embed_dim % v_dim == 0:
            self.proj_v = None

        v_dim = embed_dim
        assert v_dim == embed_dim

        self.qcrf_layer = BasicQCRFLayer(
            dim=embed_dim, depth=depth, num_heads=num_heads, v_dim=v_dim, window_size=window_size,
            mlp_ratio=4., qk_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer,
            downsample=None, use_checkpoint=False, rpe=rpe
        )

        layer = norm_layer(embed_dim)
        layer_name = 'norm_crf'
        self.add_module(layer_name, layer)

    def forward(self, x, v):
        if self.proj_x is not None:
            x = self.proj_x(x)
        if self.proj_v is not None:
            v = self.proj_v(v)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).transpose(2, 3).contiguous()
        x_out, H, W, x, Wh, Ww = self.qcrf_layer(x, v, Wh, Ww)
        norm_layer = getattr(self, f'norm_crf')
        x_out = norm_layer(x_out)
        out = x_out.view(-1, H, W, self.embed_dim).contiguous().permute(0, 3, 1, 2).contiguous()

        return out
    