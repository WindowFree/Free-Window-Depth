import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class DHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(DHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_du, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_du)))
        return act_fn(out)


class DepthRefineBlock(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=256, itrs = 4):
        super().__init__()

        self.project = nn.Conv2d(256, hidden_dim, 1, padding=0)
        self.depth_spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=context_dim//2, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=context_dim//2, out_channels=context_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=context_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        )
        self.cproj = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(context_dim, context_dim, 1)
        )
        self.itrs = itrs
        self.gru = SepConvGRU(hidden_dim=context_dim, input_dim=hidden_dim)
        self.dhead = DHead(context_dim, context_dim * 4)


    def forward(self, depth, context):
        depth_list = []
        depth_list.append(depth)
        depth = depth.detach()
        context = context.detach()
        for _ in range(self.itrs):
            depth_feature = self.depth_spatial_encoder(depth)
            context_feature = self.cproj(context)
            context = self.gru(context_feature, depth_feature)
            depth_delta = self.dhead(context)
            depth = (depth.detach() + depth_delta).clamp(1e-3, 1)
            depth_list.append(depth)

        return depth_list