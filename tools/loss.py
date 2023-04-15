import torch
import torch.nn.functional as F
from torch import nn
from models.edge_sobel import edge_for_loss
from torchvision.utils import save_image
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return focal_loss

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
            IoU1 = Iand1/(Ior1 + 1e-5)
            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        return IoU/b

class Focal_IoU(nn.Module):
    def __init__(self,theta) -> None:
        super().__init__()
        self.focal=FocalLoss()
        self.iou=IoU_loss()
        self.theta=theta
    def forward(self,pred,target):
        return self.theta*self.focal(pred,target)+(1-self.theta)*self.iou(pred,target)



class EdgeLoss_BCE(nn.Module):
    def __init__(self,device,alpha) -> None:
        super().__init__()
        self.device=device
        self.alpha=alpha
        self.bce=nn.BCELoss()
        self.mse=nn.MSELoss(reduction='mean')
        self.edge=edge_for_loss(device=self.device)
    def forward(self,yhat,y):
        bce_loss=self.bce(yhat,y)

        yall,yedge1,yedge2,yedge3,yedge4=self.edge(yhat)
        all,edge1,edge2,edge3,edge4=self.edge(y)

        edgeloss=self.mse(yedge1,edge1)+self.mse(yedge2,edge2)+self.mse(yedge3,edge3)+self.mse(yedge4,edge4)

        return (self.alpha)*edgeloss+bce_loss