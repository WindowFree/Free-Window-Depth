import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class NormalHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NormalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim//2, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_dim//2, 3, 3, padding=1)
        
    def forward(self, x, scale):
        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        x = self.conv2(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x
    
    
class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim // 2, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_dim//2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        _, _, H, W = x.shape
        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        x = self.conv2(x)
        x = self.sigmoid(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x
    
    
class ScaleHead(nn.Module):
    def __init__(self, input_dim=100):
        super(ScaleHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        
    def forward(self, x, scale):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = F.sigmoid(x) * 0.02
        if scale > 1:
            x = upsample(x, scale_factor=scale, mode="nearest")
        return x
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FocalHead(nn.Module):
    def __init__(self, input_height, input_width, input_dim=100,):
        super(FocalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim//2, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(input_dim//2, 1, 3, stride=2, padding=1)
        embed_dim = (input_height//4) * (input_width//4)
        self.mlp1 = MLP(in_features=embed_dim, hidden_features=embed_dim//4, out_features=1024)
        self.mlp2 = MLP(in_features=1024, hidden_features=64, out_features=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.sigmoid(x)
        x = x * torch.pi
        return x
        