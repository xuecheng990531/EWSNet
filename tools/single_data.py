import os
import cv2
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full")

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

class fundus_data(Dataset):
    def __init__(self, data_root: str, mode: str,name:str,isnoise:bool):
        super(fundus_data, self).__init__()
        
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.mode=mode
        self.name=name
        self.isnoise=isnoise
        self.imgs_dir = os.path.join(data_root, "img/")
        self.masks_dir=os.path.join(data_root,"mask/")

        self.image_names = [file for file in os.listdir(self.imgs_dir)]

        print(f'{mode}:dataset with {len(self.image_names)} examples.')

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image,mask, mask_thick,mask_thin):
        if self.mode == "train":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(height=512,width=512,p=1),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                A.RandomBrightnessContrast(p=0.2)
                ],additional_targets={'mask_thin':'image','mask_thick':'image','mask':'image'}
                )
            transformed = transform(image=image, mask_thick=mask_thick,mask_thin=mask_thin,mask=mask)
            image=transformed['image']
            mask_thin=transformed['mask_thin']
            mask_thick=transformed['mask_thick']
            mask=transformed['mask']
            
            return image,mask_thin,mask_thick,mask
        else:
            transform = A.Compose([
                A.Resize(height=512,width=512,p=1),
                ],additional_targets={'mask_thin':'image','mask_thick':'image','mask':'image'})
            transformed = transform(image=image, mask_thick=mask_thick,mask_thin=mask_thin,mask=mask)
            image=transformed['image']
            mask_thin=transformed['mask_thin']
            mask_thick=transformed['mask_thick']
            mask=transformed['mask']
            
            return image,mask_thin,mask_thick,mask

    def preprocess_fanet(self, image,mask, mask_thick,mask_thin,th_):
        if self.mode == "train":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.Resize(height=512,width=512,p=1),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                A.RandomBrightnessContrast(p=0.2)
                ],additional_targets={'th_':'image','mask_thin':'image','mask_thick':'image','mask':'image'}
                )
            transformed = transform(th_=th_,image=image, mask_thick=mask_thick,mask_thin=mask_thin,mask=mask)
            image=transformed['image']
            mask_thin=transformed['mask_thin']
            mask_thick=transformed['mask_thick']
            mask=transformed['mask']
            th_=transformed['th_']
            
            return image,mask_thin,mask_thick,mask,th_
        else:
            transform = A.Compose([
                # A.Resize(height=512,width=512,p=1),
                ],additional_targets={'th_':'image','mask_thin':'image','mask_thick':'image','mask':'image'})
            transformed = transform(th_=th_,image=image, mask_thick=mask_thick,mask_thin=mask_thin,mask=mask)
            image=transformed['image']
            mask_thin=transformed['mask_thin']
            mask_thick=transformed['mask_thick']
            mask=transformed['mask']
            th_=transformed['th_']
            
            return image,mask_thin,mask_thick,mask,th_


    def __getitem__(self, index):
        # 获取image和mask的路径
        
        image_name = self.image_names[index]
        if self.isnoise:
            image_path=os.path.join(self.noise_imgs_dir,image_name)
        else:
            image_path=os.path.join(self.imgs_dir,image_name)

        if self.mode=='train' or self.mode=='test':
            if self.name=='chasedb1':
                # CHASE_DB1
                mask_thick_path=os.path.join(self.masks_thick_dir,image_name.split('.')[0]+'_1stHO.png')
                mask_thin_path=os.path.join(self.masks_thin_dir,image_name.split('.')[0]+'_1stHO.png')
                mask_path = os.path.join(self.masks_dir, image_name.split('.')[0] + '_1stHO.png')
            if self.name=='stare':
                # # STARE
                mask_thick_path=os.path.join(self.masks_thick_dir,image_name.split('.')[0]+'.png')
                mask_thin_path=os.path.join(self.masks_thin_dir,image_name.split('.')[0]+'.png')
                mask_path = os.path.join(self.masks_dir, image_name.split('.')[0] +'.vk.ppm')
            if self.name=='hrf':
                # # hrf
                mask_path=os.path.join(self.masks_dir,image_name.split('.')[0]+'.tif')
                mask_thick_path=os.path.join(self.masks_thick_dir,image_name.split('.')[0]+'.png')
                mask_thin_path=os.path.join(self.masks_thin_dir,image_name.split('.')[0]+'.png')
            if self.name=='uwf':
                # uwf
                mask_path = os.path.join(self.masks_dir, image_name)
                mask_thick_path=os.path.join(self.masks_thick_dir,image_name.split('.')[0]+'.png')
                mask_thin_path=os.path.join(self.masks_thin_dir,image_name.split('.')[0]+'.png')

            assert os.path.exists(image_path), f"file '{image_path}' does not exist."
            assert os.path.exists(mask_thick_path), f"file '{mask_thick_path}' does not exist."
            assert os.path.exists(mask_thin_path), f"file '{mask_thin_path}' does not exist."
            assert os.path.exists(mask_path), f"file '{mask_path}' does not exist."

            image=cv2.imread(image_path,1)
            image=image[..., 1]
            image=cv2.resize(image,(512,512))

            mask_thin=cv2.imread(mask_thin_path,0)
            mask_thin=cv2.resize(mask_thin,(512,512))

            mask_thick=cv2.imread(mask_thick_path,0)
            mask_thick=cv2.resize(mask_thick,(512,512))

            mask=cv2.imread(mask_path,0)
            mask=cv2.resize(mask,(512,512))

            image,mask_thin,mask_thick,mask = self.preprocess(image,mask=mask, mask_thin=mask_thin,mask_thick=mask_thick)


            image=transforms.ToTensor()(image)
            mask_thin_img=transforms.ToTensor()(mask_thin)
            mask_thick_img=transforms.ToTensor()(mask_thick)
            mask = transforms.ToTensor()(mask)
            
            mask_thick_img[mask_thick_img>0.5]=1
            mask_thick_img[mask_thick_img<=0.5]=0

            mask_thin_img[mask_thin_img>0.5]=1
            mask_thin_img[mask_thin_img<=0.5]=0

            mask[mask>0.5]=1
            mask[mask <= 0.5] = 0

            return image,mask_thin_img,mask_thick_img,mask

        

if __name__ == '__main__':
    import torch
    data=fundus_data('data/hrf/train',mode='test')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2].shape)
    print(data[0][3].shape)
    print(len(data))