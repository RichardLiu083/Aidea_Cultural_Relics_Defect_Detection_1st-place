#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import random
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import DataLoader, Dataset

from Faster_RCNN.Toolbox import mixup, mosaic

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return None
seed_everything()

# In[ ]:


def read_img(img_path):
    
    img= cv2.imread(img_path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img= img/255 
    ori_img_shape= img.shape
    
    return img, ori_img_shape


# In[ ]:


class training_dataset(Dataset):
    
    def __init__(self, dataset, transforms=None, mixup= False, mosaic= False):
        
        self.dataset= dataset
        self.transforms= transforms
        self.mixup= mixup
        self.mosaic= mosaic
        
    def __getitem__(self, index):
        
        # read image and bbox
        data= self.dataset[index]
        img, ori_img_shape= read_img(data['image_path'])
        bbox, label= data['bbox'], data['label']
        
        # mosaic
        if self.mosaic and np.random.rand() >= 0.5:
            rand_img= [img.copy()]
            rand_bbox= [bbox.copy()]
            rand_label= [label.copy()]
            for i in range(3):
                choose_indx= np.random.randint(len(self.dataset))
                rand_data= self.dataset[choose_indx]
                rand_img.append( read_img(rand_data['image_path'])[0] )
                rand_bbox.append(rand_data['bbox'].copy())
                rand_label.append(rand_data['label'].copy())
                
            new_img, new_bbox, new_label= mosaic(rand_img, rand_bbox, rand_label)

            if len(new_label)<100 and len(new_label)!=0:
                img= new_img
                bbox= new_bbox
                label= new_label
        
        # mixup
        if self.mixup and np.random.rand() >= 0.5:
            choose_indx= np.random.randint(len(self.dataset))
            img_1, bbox_1, label_1= img.copy(), bbox.copy(), label.copy()
            data_2= self.dataset[choose_indx]
            bbox_2, label_2= data_2['bbox'], data_2['label']
            img_2, ori_img_shape_2= read_img(data_2['image_path'])
            
            new_img, new_bbox, new_label= mixup(img_1, bbox_1, label_1,
                                                img_2, bbox_2, label_2)
            if len(new_bbox)<=100:
                img, bbox, label= new_img, new_bbox, new_label
                
        # 轉換標籤型態
        label= torch.tensor(label, dtype=torch.int64)

        # 需製作成target字典型態
        target = {}
        target['boxes'] = bbox
        target['labels'] = label

        if self.transforms:
            while True:
                sample = {
                    'image': img,
                    'bboxes': target['boxes'],
                    'labels': label
                }
                sample = self.transforms(**sample)
                if len(sample['bboxes']) != 0: break
                    
            img = sample['image']
            
            # 經過資料增強後再轉成tensor
            target['boxes']=  torch.tensor(sample['bboxes'])

        return data['image_path'], img, target
            
    def __len__(self):
        return len(self.dataset)


# In[ ]:


class testing_dataset(Dataset):
    
    def __init__(self, dataset, transforms=None):
        
        self.dataset= dataset
        self.transforms= transforms
        
    def __getitem__(self, index):
        
        # read image
        data= self.dataset[index]
        img, ori_img_shape= read_img(data['image_path'])
        
        ori_img_shape= ori_img_shape[:2]
        augment= self.transforms(image= img)
        img= augment['image']
        
        return data['image_path'], ori_img_shape, img
            
    def __len__(self):
        return len(self.dataset)


# In[ ]:


def get_train_transform(size= 512):
    aug= [
        A.Blur(blur_limit= 3, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit= 30,
                        interpolation=cv2.INTER_LINEAR, border_mode= 0, p=0.5),
        ToTensorV2(p=1.0),
    ]
#     if size!=None:
#         aug= [A.SmallestMaxSize(max_size=size, interpolation=3, p=1)] + aug
    return A.Compose(aug, bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels']) )


def get_train_no_transform(size= 512):
    aug= [
        ToTensorV2(p=1.0),
    ]
#     if size!=None:
#         aug= [A.SmallestMaxSize(max_size=size, interpolation=3, p=1)] + aug
    return A.Compose(aug, bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels']) )


def get_test_transform(size= 512):
    aug= [
        ToTensorV2(p=1.0),
    ]
#     if size!=None:
#         aug= [A.SmallestMaxSize(max_size=size, interpolation=3, p=1)] + aug
    return A.Compose(aug)

def get_TTA_transform(flip):
    aug= [ToTensorV2(p=1.0)]
    if flip=='H':
        aug= [A.HorizontalFlip(p=1)] + aug
    elif flip=='V':
        aug= [A.VerticalFlip(p=1)] + aug
    elif flip=='HV':
        aug= [
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ] + aug
    return A.Compose(aug)

def get_TTA_revert_transform(flip):
    aug= [ToTensorV2(p=1.0)]
    if flip=='H':
        aug= [A.HorizontalFlip(p=1)] + aug
    elif flip=='V':
        aug= [A.VerticalFlip(p=1)] + aug
    elif flip=='HV':
        aug= [
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ] + aug
    return A.Compose(aug, bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels']) )