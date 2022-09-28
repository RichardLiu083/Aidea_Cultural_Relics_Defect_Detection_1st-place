#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import time
import shutil
import json
import cv2
import random

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Faster_RCNN.Data_processing import training_dataset, get_train_transform, get_train_no_transform
from Faster_RCNN.Toolbox import IOU_score, NMS, caculate_map


# In[ ]:


def valid_epoch(model, dataloader, gpu_device, CFG):
    
    model.eval()
    cpu_device = torch.device("cpu")
    
    vali_loss= []
    vali_score= []
    class_score= {}
    all_target= []
    all_pred= []
    for i, (names, images, targets) in enumerate(tqdm(dataloader)):
        
        images = [image.to(gpu_device) for image in images]
        targets = [{k: v.to(gpu_device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            """
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            vali_loss+= [loss_value]
            """
            outputs= model(images)
            outputs= [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        all_target+= targets
        
        for j in range(len(outputs)):
            
            sample = images[j].permute(1,2,0).cpu().numpy()
            
            bbox= outputs[j]['boxes'].data.cpu().numpy()
            score= outputs[j]['scores'].data.cpu().numpy()
            predicts= outputs[j]['labels'].data.cpu().numpy()

            conf= CFG['confidence']
            bbox= bbox[score>=conf].astype(np.int32)
            predicts= predicts[score>=conf]
            score= score[score>=conf]
            
            # NMS
            if CFG['NMS_threshold'] and len(bbox)!=0:
                bbox, score, predicts= NMS(sample.shape[1],
                                           sample.shape[0],
                                           bbox,
                                           score,
                                           predicts,
                                           CFG['NMS_threshold'])
            
            outputs[j]['boxes']= bbox
            outputs[j]['labels']= predicts
            outputs[j]['scores']= score
            
            metrice= IOU_score(outputs[j], targets[j], images[j].shape[1:], class_score)
            iou, class_score= metrice.caculate_IOU_score()
            vali_score+= [iou]
            
        all_pred+= outputs
            
    # caculate mean classes IOU
    for key in list(class_score.keys()):
        class_score[key]= np.mean(class_score[key])
            
    # caculate MAP
    MAP, AP_class= caculate_map(all_pred, all_target)
    
    return np.mean(vali_loss), np.mean(vali_score), class_score, MAP, AP_class


# In[ ]:


def Valid_Faster_RCNN(dataset,
                      CFG):
    
    # check gpu device
    gpu_device= 'cuda:'+CFG['device'][-1] if CFG['device'] != 'cpu' else 'cpu'
    print('using '+ gpu_device)
    
    valid_dataset= training_dataset(dataset,
                                    False,
                                    False,
                                    get_train_no_transform(size= CFG['img_size']))
   
    def collate_fn(batch):
        return tuple(zip(*batch))

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size= 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    if type(CFG['load_model'])!=str:
        print('load existing torch model')
        model= CFG['load_model']
    else:
        print('load_model: ', CFG['load_model'])
        model= torch.load(CFG['load_model'])
    model.to(gpu_device)
    
    _, valid_score, valid_class_score, MAP, AP_class= valid_epoch(model, valid_data_loader, gpu_device, CFG)
    
    print('valid_IOU:', valid_score)
    print('valid_class_IOU:', valid_class_score)
    print()
    print('valid_MAP:', MAP)
    print('valid_AP_class:', AP_class)
    
    return valid_score, valid_class_score, MAP, AP_class

