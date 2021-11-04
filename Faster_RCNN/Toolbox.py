import os
import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
from ensemble_boxes import *
from map_boxes import mean_average_precision_for_boxes
from tqdm import tqdm
from PIL import Image

def caculate_map(all_pred, all_target):
    
    pred_format= []
    target_format= []
    for i in range(len(all_pred)):
        
        all_target[i]['boxes']= all_target[i]['boxes'].numpy()
        
        for j in range(len(all_pred[i]['boxes'])):
            all_pred[i]['boxes'][j][1], all_pred[i]['boxes'][j][2]= all_pred[i]['boxes'][j][2], all_pred[i]['boxes'][j][1]
            pred_format.append( [str(i)+'.jpg'] +\
                               [str(all_pred[i]['labels'][j])] +\
                               [all_pred[i]['scores'][j]] +\
                               (all_pred[i]['boxes'][j]).tolist() )
            
        for j in range(len(all_target[i]['boxes'])):
            all_target[i]['boxes'][j][1], all_target[i]['boxes'][j][2]= all_target[i]['boxes'][j][2], all_target[i]['boxes'][j][1]
            target_format.append( [str(i)+'.jpg'] + \
                                 [str(all_target[i]['labels'][j])] + \
                                 (all_target[i]['boxes'][j]).tolist() )
            
    MAP= mean_average_precision_for_boxes(target_format, pred_format, iou_threshold=0.5, verbose= False)
    
    for i in range(len(list(MAP[1].keys()))):
        MAP[1][ list(MAP[1].keys())[i] ]= MAP[1][ list(MAP[1].keys())[i] ][0]
    
    return MAP[0], MAP[1]


def NMS(w, l, bbox, score, predicts, nms_thesh):
    
    bbox= np.array(bbox).astype(np.float)
    for j in range(len(bbox)):
        bbox[j][0]/= w
        bbox[j][1]/= l
        bbox[j][2]/= w
        bbox[j][3]/= l

    bbox, score, predicts= nms(
                            [bbox],
                            [score],
                            [predicts],
                            weights=None,
                            #method=method,
                            iou_thr= nms_thesh,
                           )

    for j in range(len(bbox)):
        bbox[j][0]*= w
        bbox[j][1]*= l
        bbox[j][2]*= w
        bbox[j][3]*= l
        
    return bbox, score, predicts


def show_predict_result(name,
                        img,
                        bbox,
                        predicts,
                        score,
                        show_img,
                        show_score,
                        show_class,
                        save_img_path):
    
    resize= 512
    origin_size= min(img.shape[:2]) if min(img.shape[:2]) > resize else resize
    
    # fix abnormally boxes
    img_shape= img.shape[:2]
    drop_indx= []
    for j in range(len(bbox)):
        if bbox[j][2]>=img_shape[1]: bbox[j][2]= img_shape[1] - 1
        if bbox[j][3]>=img_shape[0]: bbox[j][3]= img_shape[0] - 1
        if bbox[j][0]<=0: bbox[j][0]= 0
        if bbox[j][1]<=0: bbox[j][1]= 0
        if bbox[j][2]<=bbox[j][0] or bbox[j][3]<=bbox[j][1]:
            drop_indx.append(j)

    bbox= np.delete(bbox, drop_indx, axis= 0)
    predicts= np.delete(predicts, drop_indx, axis= 0)
    score= np.delete(score, drop_indx, axis= 0)
    
    # 縮小進行繪圖
    transform= A.Compose(
                    [A.SmallestMaxSize(max_size= resize, interpolation=1, p=1)],
                    bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels'])
    )
    aug= transform(image= img, bboxes= bbox, labels= np.ones(len(bbox)))
    img= aug['image']
    bbox= aug['bboxes']
    bbox= np.array(bbox).astype(np.int)
    
    for j in range(len(bbox)):
        
        cv2.rectangle(img,
                      (bbox[j][0], bbox[j][1]),
                      (bbox[j][2], bbox[j][3]),
                      (255, 0, 0), 1)

        if show_score:
            cv2.putText(img, str(score[j])[:5], (int(bbox[j][0]),int(bbox[j][3]+30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,0,0),
                        1)

        if show_class:
            cv2.putText(img, str(predicts[j]), (int(bbox[j][0]),int(bbox[j][3]+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,0,0),
                        1)

    # 還原進行儲存
    transform= A.Compose(
                    [A.SmallestMaxSize(max_size=origin_size, interpolation=1, p=1)],
    )
    aug= transform(image= img)
    img= aug['image']
    
    if show_img:
        plt.imshow(img)
        plt.show()
    
    if save_img_path != False:
        img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_img_path+'/'+name.split('/')[-1], img )
            
    return None
    


class IOU_score:
    
    def __init__(self, predict, target, img_shape, class_score):
        
        self.predict= predict
        self.target= target
        self.img_shape= img_shape
        self.class_score= class_score

    def IOU(self, mask_1, mask_2):

        mask_1= list(np.array(mask_1).reshape(-1))
        mask_2= list(np.array(mask_2).reshape(-1))

        if len(mask_1) != len(mask_2): print('shape of mask does not match')

        inter= 0
        union= 0
        for i in range(len(mask_1)):
            if mask_1[i]==1 and mask_2[i]==1:
                inter+= 1
            if mask_1[i]==1 or mask_2[i]==1:
                union+= 1
                       
        if union==0:
            iou= 1
        else:
            iou= inter/union

        return iou

    def bbox_2_mask(self, bbox, mask_shape):
        mask= np.zeros(mask_shape)
        for box in bbox:
            mask[ box[1]:box[3], box[0]:box[2] ]= 1
        return cv2.resize(mask, (256,256))

    def caculate_IOU_score(self):

        self.predict['labels']= [int(p) for p in self.predict['labels']]
        self.target['labels']= [int(p.numpy()) for p in self.target['labels']]
        all_classes= list(set(self.predict['labels']).union(self.target['labels']))
        
        # add element to class_score
        for l in list(set(self.predict['labels'])):
            if str(l) not in list(self.class_score.keys()):
                self.class_score[str(l)]= []
        for l in list(set(self.target['labels'])):
            if str(l) not in list(self.class_score.keys()):
                self.class_score[str(l)]= []

        classes_iou= []
        for label in all_classes:

            predict_bbox= [ self.predict['boxes'][i] for i in range(len(self.predict['boxes'])) if self.predict['labels'][i]==label ]
            predict_bbox= [box.astype(np.int) for box in predict_bbox]
            target_bbox= [ self.target['boxes'][i] for i in range(len(self.target['boxes'])) if self.target['labels'][i]==label ]
            target_bbox= [box.numpy().astype(np.int) for box in target_bbox]

            #print('make mask')
            predict_mask= self.bbox_2_mask(predict_bbox, self.img_shape)
            target_mask= self.bbox_2_mask(target_bbox, self.img_shape)

            #print('count iou score')
            iou_score= self.IOU(predict_mask, target_mask)
            self.class_score[str(label)].append(iou_score)
            classes_iou+= [iou_score]

        return np.mean(classes_iou), self.class_score
    
    
def check_dataset(dataset, drop_empty= False):
    print('checking dataset')
    
    drop_sample= []
    for i in tqdm(range(len(dataset))):

        data= dataset[i]
        img_shape= Image.open(data['image_path']).size
        bbox= data['bbox']

        drop_indx= []
        for j in range(len(bbox)):
            if bbox[j][2]>=img_shape[0]: bbox[j][2]= img_shape[0] - 1
            if bbox[j][3]>=img_shape[1]: bbox[j][3]= img_shape[1] - 1
            if bbox[j][0]<=0: bbox[j][0]= 0
            if bbox[j][1]<=0: bbox[j][1]= 0
            if bbox[j][2]<=bbox[j][0] or bbox[j][3]<=bbox[j][1]:
                #print('remove bbox too small')
                drop_indx.append(j)

        data['bbox']= np.delete(data['bbox'], drop_indx, axis= 0)
        data['label']= np.delete(data['label'], drop_indx, axis= 0)
        try:
            data['score']= np.delete(data['score'], drop_indx, axis= 0)
        except:
            pass

        
        if len(data['bbox'])==0 and drop_empty:
            drop_sample.append(i)

    if drop_empty:
        dataset= np.delete(dataset, drop_sample, axis= 0)
        if drop_sample!=[]: print('remove empty bboxes data: {}'.format(len(drop_sample)))
    
    return dataset


def read_yolo_format(ori_img_shape, label_path):
    
    # read label and bbox
    label= []
    bbox= []

    txt= open(label_path, 'r')
    line= txt.readline()
    while line:
        line= line.split(' ')
        line[-1]= line[-1].split('\n')[0]
        # 扣除labelimg預設15類，再+1避免成為背景類
        label.append(int(line[0])-14)
        bbox.append(np.array(line[1:]).astype(np.float32))
        line= txt.readline()

    # faster rcnn bbox格式: ( x_min, y_min, x_max, y_max )
    for i in range(len(bbox)):

        bbox[i][2]= int(bbox[i][2]*ori_img_shape[1])
        bbox[i][3]= int(bbox[i][3]*ori_img_shape[0])

        bbox[i][0]= int(bbox[i][0]*ori_img_shape[1]) - int(bbox[i][2]/2)
        bbox[i][1]= int(bbox[i][1]*ori_img_shape[0]) - int(bbox[i][3]/2)
        bbox[i][2]= bbox[i][2] + bbox[i][0]
        bbox[i][3]= bbox[i][3] + bbox[i][1]

        # fix box  
        if abs(bbox[i][0]-bbox[i][2]) < max(ori_img_shape)*0.005: bbox[i][2]+= int(max(ori_img_shape)*0.005)
        if abs(bbox[i][1]-bbox[i][3]) < max(ori_img_shape)*0.005: bbox[i][3]+= int(max(ori_img_shape)*0.005)

        # fix bbox
        for j in range(4):
            if bbox[i][j]<0: 
                bbox[i][j]= 0
            if j%2==0:
                if bbox[i][j]>ori_img_shape[1]:
                    bbox[i][j]=ori_img_shape[1]
            else:
                if bbox[i][j]>ori_img_shape[0]:
                    bbox[i][j]=ori_img_shape[0]
                                  
    return np.array(bbox), np.array(label)


def mixup(img_1, bbox_1, label_1,
          img_2, bbox_2, label_2):
    
    min_w, min_h= min(img_1.shape[1], img_2.shape[1]), min(img_1.shape[0], img_2.shape[0])
    aug= A.Compose([
            A.RandomCrop(height= min_h, width= min_w, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit= 15,
                            interpolation=cv2.INTER_LINEAR, border_mode= 0, p=0.5),
         ],
         bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels']))
    transform_1= aug(image= img_1, bboxes= bbox_1, labels= label_1)
    img_1= transform_1['image']
    bbox_1= transform_1['bboxes']
    label_1= transform_1['labels']
    transform_2= aug(image= img_2, bboxes= bbox_2, labels= label_2)
    img_2= transform_2['image']
    bbox_2= transform_2['bboxes']
    label_2= transform_2['labels']


    img= img_1*0.5 + img_2*0.5
    bbox= np.array( list(bbox_1) + list(bbox_2) )
    label= np.array( list(label_1) + list(label_2) )

    return img, bbox, label


def mosaic(all_img,      #[img_1, img_2, img_3, img_4]
           all_bbox,     #[bbox_1, bbox_2, bbox_3, bbox_4]
           all_label):   #[label_1, label_2, label_3, label_4]
    
    width, height= all_img[0].shape[1], all_img[0].shape[0]
    
    center_point_x= np.random.randint( int(width*0.25), int(width*0.75) )
    center_point_y= np.random.randint( int(height*0.25), int(height*0.75) )
    
    crop_area_w_h= [(center_point_x, center_point_y),
                    (width-center_point_x, center_point_y),
                    (center_point_x, height-center_point_y),
                    (width-center_point_x, height-center_point_y)]
    
    background_img= np.zeros_like(all_img[0])
    new_bbox= []
    new_label= []
    for i in range(len(all_img)):
        img= np.array(all_img[i])
        bbox= all_bbox[i]
        label= all_label[i]
        
        cut_aug= A.Compose([A.PadIfNeeded(min_height= crop_area_w_h[i][1], min_width=crop_area_w_h[i][0], border_mode=0, p=1),
                            A.RandomCrop(height= crop_area_w_h[i][1], width= crop_area_w_h[i][0], p=1)],
                       bbox_params=A.BboxParams(format= 'pascal_voc', label_fields=['labels']))
        mosaic_aug= cut_aug(image= img, bboxes= bbox, labels= label)
        
        img= np.array(mosaic_aug['image'])
        bbox= np.array(mosaic_aug['bboxes'])
        label= np.array(mosaic_aug['labels'])
        
        if i==0: 
            background_img[:center_point_y, :center_point_x]= img
        elif i==1: 
            background_img[:center_point_y, center_point_x:width]= img
            if len(bbox)!=0:
                bbox[:,0]+= center_point_x
                bbox[:,2]+= center_point_x
        elif i==2: 
            background_img[center_point_y:height, :center_point_x]= img
            if len(bbox)!=0:
                bbox[:,1]+= center_point_y
                bbox[:,3]+= center_point_y
        elif i==3: 
            background_img[center_point_y:height, center_point_x:width]= img
            if len(bbox)!=0:
                bbox[:,0]+= center_point_x
                bbox[:,2]+= center_point_x
                bbox[:,1]+= center_point_y
                bbox[:,3]+= center_point_y
            
        new_bbox+= list(bbox)
        new_label+= list(label)
        
    new_img= background_img 
    new_bbox= np.array(new_bbox)
    new_label= np.array(new_label)
    
    # check bbox
    img_shape= [new_img.shape[1], new_img.shape[0]]
    drop_indx= []
    for j in range(len(new_bbox)):
        if new_bbox[j][2]>=img_shape[0]: new_bbox[j][2]= img_shape[0] - 1
        if new_bbox[j][3]>=img_shape[1]: new_bbox[j][3]= img_shape[1] - 1
        if new_bbox[j][0]<=0: new_bbox[j][0]= 0
        if new_bbox[j][1]<=0: new_bbox[j][1]= 0
        if new_bbox[j][2]<=new_bbox[j][0]+int(img_shape[0]*0.02) or new_bbox[j][3]<=new_bbox[j][1]+int(img_shape[1]*0.02):
            drop_indx.append(j)

    new_bbox= np.delete(new_bbox, drop_indx, axis= 0)
    new_label= np.delete(new_label, drop_indx, axis= 0)
        
    return new_img, np.array(new_bbox), np.array(new_label)