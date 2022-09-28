#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import albumentations as A
import cv2
import shutil
import random
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from ensemble_boxes import *
from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from Faster_RCNN.Data_processing import testing_dataset, get_test_transform, get_TTA_transform, get_TTA_revert_transform
from Faster_RCNN.Toolbox import show_predict_result, NMS, check_dataset

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return None
seed_everything()



def test_epoch(model, test_data_loader, gpu_device, CFG):
    
    model.eval()
    
    test_result= []
    for name, ori_size ,images in tqdm(test_data_loader):
        
        images = list(img.to(gpu_device) for img in images)

        cpu_device = torch.device("cpu")

        with torch.no_grad():
            outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # read every result in batch
        for i in range(len(outputs)):
            
            result= {
                'image_path': None,
                'bbox': None,
                'label': None,
            }

            sample = images[i].permute(1,2,0).cpu().numpy()

            bbox= outputs[i]['boxes'].data.cpu().numpy()
            score= outputs[i]['scores'].data.cpu().numpy()
            predicts= outputs[i]['labels'].data.cpu().numpy()
            
            # check bbox area
            area= (bbox[:,2]-bbox[:,0]) * (bbox[:,3]-bbox[:,1])
            zero_area= (area==0).astype(np.int).sum()
            
            # NMS
            if CFG['NMS_threshold'] and len(bbox)!=0 and zero_area==0:
                bbox, score, predicts= NMS(sample.shape[1],
                                           sample.shape[0],
                                           bbox,
                                           score,
                                           predicts,
                                           CFG['NMS_threshold'])
            
            # 還原圖片及其bbox
            sample= cv2.resize( sample, tuple(reversed(ori_size[i])) )
            for j in range(len(bbox)):
                bbox[j][0]= int( bbox[j][0]*(ori_size[i][1]/images[i].shape[2]) )
                bbox[j][1]= int( bbox[j][1]*(ori_size[i][0]/images[i].shape[1]) )
                bbox[j][2]= int( bbox[j][2]*(ori_size[i][1]/images[i].shape[2]) )
                bbox[j][3]= int( bbox[j][3]*(ori_size[i][0]/images[i].shape[1]) )
            
            # add result
            result['image_path']= name[i]
            result['bbox']= bbox
            result['label']= predicts
            result['score']= score
            test_result.append(result)
            
    return test_result



def WBF(test_result, all_result):
    
    for i in range(len(all_result[0])):
        all_bbox= []
        all_label= []
        all_score= []
        for j in range(len(all_result)):

            bbox, score, label= all_result[j][i]['bbox'], all_result[j][i]['score'], all_result[j][i]['label']
            w, l= Image.open(all_result[j][i]['image_path']).size

            bbox= np.array(bbox).astype(np.float16)
            for k in range(len(bbox)):
                bbox[k][0]/= w
                bbox[k][1]/= l
                bbox[k][2]/= w
                bbox[k][3]/= l

            all_bbox.append(bbox)
            all_label.append(label)
            all_score.append(score)

        # WBF
        bbox, score, label= weighted_boxes_fusion(all_bbox,
                                                  all_score,
                                                  all_label,
                                                  weights= None,
                                                  #method=method,
                                                  iou_thr=0.57,
                                                  skip_box_thr= 0.001)

        for k in range(len(bbox)):
            bbox[k][0]*= w
            bbox[k][1]*= l
            bbox[k][2]*= w
            bbox[k][3]*= l

        test_result[i]['bbox']= bbox.astype(np.int)
        test_result[i]['label']= label
        test_result[i]['score']= score.astype(np.float)
        
    return test_result



def TTA(model, TTA_data_loader, gpu_device, CFG, flip):
    TTA_result= test_epoch(model, TTA_data_loader, gpu_device, CFG)
    TTA_result= check_dataset(TTA_result, drop_empty= False)
    TTA_transform= get_TTA_revert_transform(flip)
    for i in range(len(TTA_result)):
        img_shape= Image.open(TTA_result[i]['image_path']).size[::-1]
        transform= TTA_transform(image= np.zeros(img_shape), bboxes= TTA_result[i]['bbox'], labels= TTA_result[i]['label'])
        TTA_result[i]['bbox']= np.array(transform['bboxes']).astype(np.int)
        TTA_result[i]['label']= transform['labels']
    return TTA_result

# In[ ]:


def Test_Faster_RCNN(dataset,
                     CFG):
    
    # check gpu device
    gpu_device= 'cuda:'+CFG['device'][-1] if CFG['device'] != 'cpu' else 'cpu'
    print('using '+ gpu_device)
    
    # read test_data
    test_dataset= testing_dataset(dataset.copy(), get_test_transform(size= CFG['img_size']))
    TTA_H_flip_dataset= testing_dataset(dataset.copy(), get_TTA_transform(flip= 'H'))
    TTA_V_flip_dataset= testing_dataset(dataset.copy(), get_TTA_transform(flip= 'V'))
    TTA_HV_flip_dataset= testing_dataset(dataset.copy(), get_TTA_transform(flip= 'HV'))

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_data_loader = DataLoader(
        test_dataset,
        batch_size= 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    TTA_H_flip_data_loader = DataLoader(
        TTA_H_flip_dataset,
        batch_size= 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    TTA_V_flip_data_loader = DataLoader(
        TTA_V_flip_dataset,
        batch_size= 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    TTA_HV_flip_data_loader = DataLoader(
        TTA_HV_flip_dataset,
        batch_size= 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # load model
    if type(CFG['load_model'])==str:
        if CFG['device']=='cpu':
            model= torch.load(CFG['load_model'], map_location='cpu')
        else:
            model= torch.load(CFG['load_model'])
    else:
        model= CFG['load_model']
        
        
    # starting testing
    if type(CFG['img_size'])!=list:
        ## single scale testing
        # change model input size
        min_edge_size= CFG['img_size']
        
        # set image size, else use default
        if min_edge_size:
            model.transform= GeneralizedRCNNTransform(min_size= int(min_edge_size),
                                                      max_size= int(min_edge_size*10),
                                                      image_mean=[0.485, 0.456, 0.406],
                                                      image_std=[0.229, 0.224, 0.225])
        model.to(gpu_device)
        
        test_result= test_epoch(model, test_data_loader, gpu_device, CFG)
        
        if CFG['TTA']:
            all_result= []
            all_result.append(test_result)
            
            TTA_result= TTA(model, TTA_H_flip_data_loader, gpu_device, CFG, flip= 'H')
            all_result.append(TTA_result)
            TTA_result= TTA(model, TTA_V_flip_data_loader, gpu_device, CFG, flip= 'V')
            all_result.append(TTA_result)
            TTA_result= TTA(model, TTA_HV_flip_data_loader, gpu_device, CFG, flip= 'HV')
            all_result.append(TTA_result)

            test_result= WBF(test_result, all_result)
    else:
        ## multi scale testing
        all_result= []
        for img_size in CFG['img_size']:
            # change model input size
            min_edge_size= img_size
            model.transform= GeneralizedRCNNTransform(min_size= int(min_edge_size),
                                                      max_size= int(min_edge_size*10),
                                                      image_mean=[0.485, 0.456, 0.406],
                                                      image_std=[0.229, 0.224, 0.225])
            model.to(gpu_device)

            test_result= test_epoch(model, test_data_loader, gpu_device, CFG)
            all_result.append(test_result)
            
            if CFG['TTA']:
                TTA_result= TTA(model, TTA_H_flip_data_loader, gpu_device, CFG, flip= 'H')
                all_result.append(TTA_result)
                TTA_result= TTA(model, TTA_V_flip_data_loader, gpu_device, CFG, flip= 'V')
                all_result.append(TTA_result)
                TTA_result= TTA(model, TTA_HV_flip_data_loader, gpu_device, CFG, flip= 'HV')
                all_result.append(TTA_result)
                
        test_result= WBF(test_result, all_result)
    
    # postprocessing
    for i, pred in enumerate(tqdm(test_result)):

        predicts= pred['label']
        bbox= pred['bbox']
        score= pred['score']

        conf= CFG['confidence']
        bbox= bbox[score>=conf].astype(np.int)
        predicts= predicts[score>=conf].astype(np.int)
        score= score[score>=conf]
        
        test_result[i]['bbox']= bbox
        test_result[i]['label']= predicts
        test_result[i]['score']= score

        # mapping classes
        if CFG['class_mapping']:
            class_map= dict((v,k) for k,v in CFG['class_mapping'].items())
            predicts= [class_map[p] for p in predicts]
            pred['label']= predicts

        # show img
        if CFG['show_img'] or CFG['save_img_path']:
            sample= cv2.imread(pred['image_path'])
            sample= cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

            show_predict_result(pred['image_path'], 
                                sample, 
                                bbox, 
                                predicts, 
                                score,
                                CFG['show_img'],
                                CFG['show_score'], 
                                CFG['show_classes'], 
                                CFG['save_img_path'])
            
    return test_result