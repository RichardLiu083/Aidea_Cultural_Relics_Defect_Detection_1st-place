#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import time
import shutil
import json
import cv2
import random

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Faster_RCNN.Data_processing import training_dataset, get_train_transform, get_train_no_transform
from Faster_RCNN.Toolbox import IOU_score
from Faster_RCNN.Valid import valid_epoch

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


def train_epoch(model, dataloader, optimizer, batch_size, gpu_device):
    
    model.train()
    
    training_loss= []
    scaler = GradScaler()
    for i, (names, images, targets) in enumerate(tqdm(dataloader)):

        images = [image.to(gpu_device) for image in images]
        targets = [{k: v.to(gpu_device) for k, v in t.items()} for t in targets]

#         with autocast():
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        training_loss.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
#         scaler.scale(losses).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
    return np.mean(training_loss)


# In[ ]:


def Train_Faster_RCNN(dataset,
                      CFG):
    # 防呆
    if CFG['epochs'] < 1: CFG['epochs']= 1
    if CFG['valid_size'] < 0: CFG['valid_size']= 0
    if CFG['batch_size'] < 1: CFG['batch_size']= 1
    CFG['img_size']= int(CFG['img_size'])
    
    # check gpu device
    gpu_device= 'cuda:'+CFG['device'][-1] if CFG['device'] != 'cpu' else 'cpu'
    print('using '+ gpu_device)
    
    # check dataset
    print(f'find {len(dataset)} image and bbox')
    
    # verify num_class
    all_class= []
    for data in dataset:
        label= list(data['label'])
        all_class+= label
    num_class= len(list(set(all_class)))
    max_class= max(list(set(all_class)))
    print('find number of classes: ', num_class)
    print('maximun number of classes:', max_class)
    
    # 打亂資料
    dataset= np.array(dataset)
    index= list(range(len(dataset)))
    np.random.shuffle(index)
    dataset= dataset[index]
    
    # 劃分訓練、驗證資料
    if CFG['valid_size'] != 0:
        train_index, valid_index= train_test_split(list(range(len(dataset))), test_size= CFG['valid_size'], random_state=0)
        train_data= dataset[train_index].copy()
        valid_data= dataset[valid_index].copy()
    else:
        train_data= dataset.copy()
        valid_data= {}
    
    # 建立資料集
    if CFG['data_aug']:
        train_dataset= training_dataset(train_data,
                                        get_train_transform(size= CFG['img_size']),
                                        mixup= CFG['mixup'],
                                        mosaic= CFG['mosaic'])
    else:
        train_dataset= training_dataset(train_data,
                                        get_train_no_transform(size= CFG['img_size']),
                                        mixup= CFG['mixup'],
                                        mosaic= CFG['mosaic'])
        
    valid_dataset= training_dataset(valid_data,
                                    get_train_no_transform(size= CFG['img_size']),
                                    mixup= False,
                                    mosaic= False)
    print(f'train on {len(train_dataset)} samples, validation on {len(valid_dataset)} samples')
    
    # build dataloader
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=  CFG['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=  CFG['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # build model
    if CFG['load_model'] != False:
        if type(CFG['load_model'])!=str:
            print('load existing torch model')
            model= CFG['load_model']
        else:
            print('load_model: ', CFG['load_model'])
            if CFG['device']=='cpu':
                model= torch.load(CFG['load_model'], map_location='cpu')
            else:
                model= torch.load(CFG['load_model'])
    else:
        # set anchor parameter
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', 'pool'],
                                                            output_size=7,
                                                            sampling_ratio=2)
        anchor_sizes= np.array( ((8,), (16,), (32,), (64,), (128,)) ) * int(CFG['anchor_ratio'])
#         anchor_sizes= np.array( ((8,16,), (32,48,), (64,96,), (128,192,), (256,384,)) ) * int(CFG['anchor_ratio'])
        anchor_sizes= tuple(map(tuple, anchor_sizes))
        aspect_ratios= ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator= AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # choose pretrained
        if CFG['pretrained']:
            backbone= resnet_fpn_backbone(CFG['model_backbone'], pretrained=True, trainable_layers= 5, returned_layers=[1,2,3,4])
        else:
            backbone= resnet_fpn_backbone(CFG['model_backbone'], pretrained=False, trainable_layers= 5, returned_layers=[1,2,3,4])
        
        # choose model architecture
        if CFG['model_architecture']=='FasterRCNN':
            model = FasterRCNN(backbone,
#                                min_size= int(min_edge_size), # 先縮放成min_size，再依據max_size決定是否縮放
#                                max_size= int(min_edge_size*10),
#                                image_mean= [0.485, 0.456, 0.406],
#                                image_std= [0.229, 0.224, 0.225],
                               rpn_anchor_generator= anchor_generator,
                               box_roi_pool=roi_pooler,
                               box_detections_per_img= 100,
                               num_classes= int(max_class+1) )  # num_classes + background
        elif CFG['model_architecture']=='RetinaNet':
            model = RetinaNet(backbone,
#                               min_size= int(min_edge_size), # 先縮放成min_size，再依據max_size決定是否縮放
#                               max_size= int(min_edge_size*10),
#                               image_mean= [0.485, 0.456, 0.406],
#                               image_std= [0.229, 0.224, 0.225],
                              anchor_generator= anchor_generator,
                              detections_per_img=300,
                              num_classes= int(max_class+1) )  # num_classes + background
            
    # change preprocessing bloch
    min_edge_size= CFG['img_size']
    model.transform= GeneralizedRCNNTransform(min_size= int(min_edge_size),
                                              max_size= int(min_edge_size*10),
                                              image_mean=[0.485, 0.456, 0.406],
                                              image_std=[0.229, 0.224, 0.225])
    model.to(gpu_device)
    
    # hyperparameter
    params = [p for p in model.parameters() if p.requires_grad]
    if CFG['optimizer']=='adam':
        optimizer = torch.optim.AdamW(params, lr= CFG['lr'])
    elif CFG['optimizer']=='sgd':
        optimizer = torch.optim.SGD(params, lr= CFG['lr'])
    elif CFG['optimizer']=='rmsprop':
        optimizer = torch.optim.RMSprop(params, lr= CFG['lr'])
    """
    lr_scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    'max',
                                                    factor= 0.1, 
                                                    patience= 4, 
                                                    min_lr= 1e-6,
                                                    verbose=True)
                                                    """
    lr_scheduler= None
    
    # 開始訓練
    print()
    train_epoch_loss= []
    train_epoch_score= []
    valid_epoch_score= []
    best_valid_score= 0
    for epoch in range(CFG['epochs']):
       
        print('epoch: '+ str(epoch))
        
        # training
        train_loss= train_epoch(model, train_data_loader, optimizer, CFG['batch_size'], gpu_device)
        train_epoch_loss.append(train_loss)
        
        # verify train dataset
        vali_CFG= {'img_size': CFG['img_size'], 'confidence': 0.5, 'NMS_threshold': 0.5}
        if CFG['valid_train_dataset']:
            try:
                _, train_iou, train_class_iou, train_MAP, train_class_MAP= valid_epoch(model, train_data_loader, gpu_device, vali_CFG)
            except:
                print('error happend when validation')
                train_iou, train_class_iou, train_MAP, train_class_MAP= 0, 0, 0, 0
                
        # verify valid dataset
        if CFG['valid_size'] != 0:
            try:
                _, valid_iou, valid_class_iou, valid_MAP, valid_class_MAP= valid_epoch(model, valid_data_loader, gpu_device, vali_CFG)
            except:
                print('error happend when validation')
                valid_iou, valid_class_iou, valid_MAP, valid_class_MAP= 0, 0, 0, 0
        else:
            valid_score, valid_class_score= 'Nan', 'Nan'
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        #儲存最佳模型-------------------------
        if CFG['save_model_path_and_name']:
            if CFG['save_best_model']== True and CFG['valid_size'] != 0:
                if valid_MAP >= best_valid_score or epoch==0:
                    best_valid_score= valid_MAP
                    torch.save(model, CFG['save_model_path_and_name'])
                    print(f'model save at score: {best_valid_score}')
            else:
                torch.save(model, CFG['save_model_path_and_name'])
        #------------------------------------
        print('train_loss: ', train_loss)
        if CFG['valid_train_dataset']:
            print('train_iou: ', train_iou)
            print('train_class_iou:', train_class_iou)
            print('train_MAP: ', train_MAP)
            print('train_class_MAP:', train_class_MAP)
        print()
        if CFG['valid_size'] != 0:
            print('valid_iou: ', valid_iou)
            print('valid_class_iou:', valid_class_iou)
            print('valid_MAP: ', valid_MAP)
            print('valid_class_MAP:', valid_class_MAP)
            print()

    return model