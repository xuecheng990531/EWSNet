import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AvgPool2d(1)
        self.max_pool = nn.MaxPool2d(1)

        self.op=nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        avg_out = self.op(self.avg_pool(x))
        max_out = self.op(self.max_pool(x))
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
    def __init__(self, in_planes, ratio=8):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.pixel_attention = PixelAttention(in_planes, ratio)

    def forward(self, x):
        x_ca = self.channel_attention(x) * x
        x_pa = self.pixel_attention(x) * x
        out = (x_ca + x_pa)
        return out


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
    


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net



class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)



class ResUnet(nn.Module):
    def __init__(self, channel,device, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.device=device
        self.maxpool = nn.MaxPool2d(2, 2)
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

        # --------------------------------------MSP Module-------------------------------------------
        self.conv1_aspp=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv2_aspp=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )


        self.conv3_aspp=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.reduce=nn.Sequential(
            nn.Conv2d(in_channels=512*4,out_channels=512*2,kernel_size=1),
            nn.BatchNorm2d(512*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512*2,out_channels=512,kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.msp=ASPP(in_channel=512,depth=512)
        # --------------------------------------MSP Module-------------------------------------------


    def forward(self, x):


        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)#torch.Size([1, 64, 256, 256])
        x1_msp=self.conv1_aspp(x1)#torch.Size([1, 512, 32, 32])


        x2 = self.residual_conv_1(x1)#torch.Size([1, 128, 128, 128])
        x2_msp=self.conv2_aspp(x2)#torch.Size([1, 512, 32, 32])


        x3 = self.residual_conv_2(x2)#torch.Size([1, 256, 64, 64])
        x3_msp=self.conv3_aspp(x3)#torch.Size([1, 512, 32, 32])

        x4 = self.bridge(x3)#torch.Size([1, 512, 32, 32])
        # x4=self.upsample_1(x4)

        x4 = self.upsample_1(self.msp(self.reduce(torch.cat([x1_msp,x2_msp,x3_msp,x4],dim=1))))#torch.Size([1, 512, 64, 64])
        x5 = torch.cat([x4, AttentionModule(in_planes=x3.shape[1]).to(self.device)(x3)], dim=1)
        # x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)
        x6 = self.upsample_2(x6)#torch.Size([1, 256, 128, 128])
        x7 = torch.cat([x6, AttentionModule(in_planes=x2.shape[1]).to(self.device)(x2)], dim=1)
        # x7 = torch.cat([x6,x2],dim=1)
        x8 = self.up_residual_conv2(x7)
        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, AttentionModule(in_planes=x1.shape[1]).to(self.device)(x1)], dim=1)
        # x9 = torch.cat([x8,x1],dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
