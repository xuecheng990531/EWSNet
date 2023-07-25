from torch import nn
from .dwt_module import *
from .dwtlayer import *
from .cbam import CBAM

class wa_module(nn.Module):
    def __init__(self, device,wavename='haar'):
        super(wa_module, self).__init__()
        self.dev=device
        self.dwt = DWT_2D(wavename=wavename,device=self.dev)
        self.softmax = nn.Softmax2d()
        

    @staticmethod
    def get_module_name():
        return "wa"

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        output = LL
        assert(len(input.shape)==4)
        
        LL=CBAM(channel_in=input.shape[1]).to(self.dev)(LL).to(self.dev)

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output, LL