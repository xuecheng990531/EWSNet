import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class PixelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes // ratio, 3, padding=1, groups=in_planes // ratio, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes // ratio)
        self.conv3 = nn.Conv2d(in_planes // ratio, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        out = self.sigmoid(x)
        out = out * avg_out
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.pixel_attention = PixelAttention(in_planes, ratio)

    def forward(self, x):
        x_ca = self.channel_attention(x) * x
        x_pa = self.pixel_attention(x) * x
        out = (x_ca + x_pa)
        return out
