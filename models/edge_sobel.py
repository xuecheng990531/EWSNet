import torch
torch.set_printoptions(profile='full')
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


def Gedge_map(im,device):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device)
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).to(device)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).to(device)
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).to(device)
    edge_detect3 = torch.abs(conv_op3(Variable(im)))

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    # 返回所有的边缘
    return sobel_out,edge_detect,edge_detect1,edge_detect2,edge_detect3


class gt_edge(nn.Module):
    def __init__(self,device):
        super(gt_edge, self).__init__()
        self.sig=nn.Sigmoid()
        self.dev=device

    def forward(self, x):
        result,_,_,_,_=self.sig(Gedge_map(x,device=self.dev))
        result[result<=0.5]=0
        result[result>0.5]=1
        print(result)
        return result
    


class img_edge(nn.Module):
    def __init__(self,device):
        super(img_edge, self).__init__()
        self.dev=device
    def forward(self, x):
        result,_,_,_,_=Gedge_map(x,self.dev)
        return result
    

class edge_for_loss(nn.Module):
    def __init__(self,device):
        super(edge_for_loss, self).__init__()
        self.dev=device
    def forward(self, x):
        all,edge1,edge2,edge3,edge4=Gedge_map(x,self.dev)
        return all,edge1,edge2,edge3,edge4
    


if __name__=='__main__':
    import cv2
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image
    img=cv2.imread('data/train/img/Image_01L.jpg',0)
    img=cv2.resize(img,(512,512))
    img=ToTensor()(img)
    img=torch.unsqueeze(img,0)

    all,edge1,edge2,edge3,edge4=edge_for_loss(device='cpu')(img)
    save_image(all,'new1.png')