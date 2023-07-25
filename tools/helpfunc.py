import torch
import numpy as np
import random
import os 
import cv2
import os
from PIL import Image

def save_checkpoint(root, model_thin,model_thick,refine_model, better,dataname):
    if better:
        fpath_thin = os.path.join(root+'/'+dataname, '{}_thin.pth'.format(dataname))
        fpath_thick = os.path.join(root+'/'+dataname, '{}_thick.pth'.format(dataname))
        fpath_refine = os.path.join(root+'/'+dataname, '{}_refine.pth'.format(dataname))
        torch.save(model_thin.state_dict(), fpath_thin)
        torch.save(model_thick.state_dict(), fpath_refine)
        torch.save(refine_model.state_dict(), fpath_thick)
    else:
        print('')
    


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def acc(final_mask_clone,labels,test_loader,pixelwize_rank):
    if final_mask_clone.shape == labels.shape:
        pixelwize_rank += int(
            torch.sum((final_mask_clone > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(test_loader)
    else:
        pixelwize_rank += int(
            torch.sum((final_mask_clone.argmax(dim=1) > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(test_loader)



def extract(label_root):
    for i in range(len(os.listdir(label_root))):
        name=os.path.join(label_root,os.listdir(label_root)[i])
        img_name=os.listdir(label_root)[i]
        img_array=cv2.imread(name,0)
        _, binary = cv2.threshold(img_array, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        thick_vessels = cv2.bitwise_and(img_array, opened)
        thin_vessels = cv2.bitwise_and(img_array, cv2.bitwise_not(opened))
        cv2.imwrite('data/drive/train/thick_mask/{}'.format(img_name.split('.')[0]+'.png'),thick_vessels)
        cv2.imwrite('data/drive/train/thin_mask/{}'.format(img_name.split('.')[0]+'.png'),thin_vessels)

        

if __name__=='__main__':
    extract('data/drive/train/mask')