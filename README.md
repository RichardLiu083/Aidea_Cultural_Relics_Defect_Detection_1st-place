# Aidea_Cultural_Relics_Defect_Detection_1st_place
**2021/11/04 - Aidea - 國立臺灣文學館典藏文物劣化辨識 - 1st place solution**  
[Competition Link](https://aidea-web.tw/topic/fbbb5b7e-4dc8-4827-974f-51a1ee725012)  
![image](https://github.com/RichardLiu083/Aidea_Cultural_Relics_Defect_Detection_1st-place/blob/main/img/Rank.png)

## Mission
![image](https://github.com/RichardLiu083/Aidea_Cultural_Relics_Defect_Detection_1st-place/blob/main/img/Mission.png)

## Insight
- 針對水漬類別瑕疵單獨建立模型。(訓練仍使用5類，驗證時只取水漬類分數最高)
- 提升訓練圖片解析度可提升模型抓取小面積瑕疵效果。(解析度大於1500後則模型表現無法再有顯著提升)
- 縮小模型中的anchor size對於抓取小面積瑕疵有顯著效果。
- 對預先釋出之測試資料集進行Pseudo Label，再加入訓練集中可提升模型效果。
- 水漬類別座標框總數為5類中最少。
- 水漬類別座標框平均比其餘類別較大。
<img src="https://github.com/RichardLiu083/Aidea_Cultural_Relics_Defect_Detection_1st-place/blob/main/img/Bbox_size.png" width="700">

## Model
<img src="https://github.com/RichardLiu083/Aidea_Cultural_Relics_Defect_Detection_1st-place/blob/main/img/Model.png" width="700">

## Augmentation
- 水平、垂直翻轉
- 隨機縮放圖像大小
- 隨機旋轉(30 degree)
- 隨機亮度對比度調整
- Mixup

## Training
- Faster_RCNN 訓練圖片解析度 1536
- Cascade_RCNN 訓練圖片解析度 1024
- Cascade_RCNN 使用可變形卷積 (DCN)。
- lr 3e-5

## Validation
- 將單一圖片中座標框數量大於150之圖片當作驗證資料集，其餘用作訓練。(可避免訓練時GPU OOM)
- 只儲存最佳驗證分數模型。

## Inference
- Faster_RCNN 預測水漬以外類別，Cascade_RCNN 單獨預測水漬類別。
- Multi-scale testing
- TTA (水平、垂直)
- WBF
