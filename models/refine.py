
import torch
import torch.nn as nn
from torch.nn import functional as F
from dwt.wlcbam import wa_module
from tools.visual import show_feature_map
# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in , C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 基于小波变换的下采样技术
class DownSampling_wa(nn.Module):
    def __init__(self, C,device):
        super(DownSampling_wa, self).__init__()
        self.dev=device
        self.wa=wa_module(device=self.dev)
        self.Up = nn.Conv2d(C, C // 2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        down_fea,skip_fea=self.wa(x)[0],self.wa(x)[1]
        origin_down=self.pool(x)
        # if x.shape[1]==64:
        #     show_feature_map(feature_map=down_fea,filename='downed_feature')
        #     show_feature_map(feature_map=skip_fea,filename='skip_feature')
        #     show_feature_map(feature_map=x,filename='origin')
        #     show_feature_map(feature_map=origin_down,filename='origin_down')
        return down_fea,skip_fea



# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


# 主干网络
class denosing_module(nn.Module):

    def __init__(self,device,inchannel):
        super(denosing_module, self).__init__()
        self.dev=device
        # 4次下采样
        self.C1 = Conv(inchannel, 64)
        self.D1 = DownSampling_wa(64,device=self.dev)
        # self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling_wa(128,device=self.dev)
        # self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling_wa(256,device=self.dev)
        # self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)


    # 这是采用小波变换模块的前向传播
    def forward(self, x):
        R1 = self.C1(x)#torch.Size([1, 64, 512, 512])
        R2 = self.C2(self.D1(R1)[0])#torch.Size([1, 128, 256, 256])
        R3 = self.C3(self.D2(R2)[0])#torch.Size([1, 256, 128, 128])
        R4 = self.C4(self.D3(R3)[0])#torch.Size([1, 512, 64, 64])

        # skip feature是小波变换的底层信号
        O2 = self.C7(self.U2(R4,self.C3(self.D2(R2)[1])))
        O3 = self.C8(self.U3(O2,self.C2(self.D1(R1)[1])))
        O4 = self.C9(self.U4(O3,R1))

        ## 没有小波变换的skip feature
        # O2 = self.C7(self.U2(R4, R3))
        # O3 = self.C8(self.U3(O2, R2))
        # O4 = self.C9(self.U4(O3, R1))

        return self.Th(self.pred(O4))

    # def forward(self, x):
    #     R1 = self.C1(x)
    #     R2 = self.C2(self.D1(R1))
    #     R3 = self.C3(self.D2(R2))
    #     R4 = self.C4(self.D3(R3))

    #     O2 = self.C7(self.U2(R4, R3))
    #     O3 = self.C8(self.U3(O2, R2))
    #     O4 = self.C9(self.U4(O3, R1))

    #     return self.Th(self.pred(O4))
    
if __name__=='__main__':
    x=torch.randn(1,1,512,512)
    model=denosing_module(device='cpu',inchannel=1)(x)
    print(model)
    print(model.shape)