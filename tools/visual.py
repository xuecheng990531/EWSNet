import torch
from torchvision.utils import save_image

def show_feature_map(feature_map,filename):
    assert len(feature_map.shape)==4,'the dim of the input feature not equal to 4'
    #以下4行，通过双线性插值的方式改变保存图像的大小
    upsample = torch.nn.UpsamplingBilinear2d(size=(512,512))
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    for index in range(6):#通过遍历的方式，将64个通道的tensor拿出
        save_image(feature_map[index - 1],"visual_output/features/"+str(filename)+'/'+str(index) + ".png")