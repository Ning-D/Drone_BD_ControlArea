from __future__ import print_function, division
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import pandas as pd
cudnn.benchmark = True
plt.ion()   # interactive mode
import natsort
import glob


# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    
    return buf

# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理） 
# sample execution (requires torchvision)






# 利用torch提供的Dataset类，定义我们自己的数据集
class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(pd.read_csv('Train_samples.csv', header=None))


    def __getitem__(self, idx):
    
    
        img_name = pd.read_csv('Train_samples.csv', header=None).loc[idx][0]

        #print('../CA/data_all/map0/' +img_name)
        #imgZ = cv2.imread('../CA/data_all/map0/' +img_name)
       
        #imgZ=imgZ.reshape(56,112,3)        
        
        #imgZ = imgZ/255

        #Z_image = imgZ
        #Z_image = (Z_image - Z_image.min()) / (Z_image.max()- Z_image.min())         
        
        
        
        
        
        img4p = pd.read_csv('../CA/data_all/map_4players/' +img_name, header=None)
        img4p=np.array(img4p)
        
        
        loc4px=img4p[:,0]
        loc4py=img4p[:,1]
                
        
        
        
        
        imgA = pd.read_csv('../CA/data_all/map2/' +img_name, header=None)
        imgA=np.array(imgA)
        imgA=imgA.reshape(56,112,1)
        
        
        
        
        
        imgB = pd.read_csv('../CA/data_all/map3/' +img_name, header=None)
        imgB=np.array(imgB)
        Rx,Ry=np.where(imgB==1)
        
        imgBB = pd.read_csv('../CA/data_all/map4/' +img_name, header=None)
        imgBB=np.array(imgBB)
        Sx,Sy=np.where(imgBB==1)
        
        
        
        
        
        imgC = pd.read_csv('../CA/data_all/map9/' +img_name, header=None)        
        imgC=np.array(imgC)/10.0 
        imgC=imgC.reshape(56,112,1)
        
        
              
        imgD = pd.read_csv('../CA/data_all/map10/' +img_name, header=None)
        imgD=np.array(imgD)/10.0
        imgD=imgD.reshape(56,112,1)
       
       
        imgBOX_h = pd.read_csv('../CA/data_all/map13/' +img_name, header=None)
        imgBOX_h=np.array(imgBOX_h)
        imgBOX_h=imgBOX_h.reshape(56,112,1)       
       
        imgBOX_w = pd.read_csv('../CA/data_all/map14/' +img_name, header=None)
        imgBOX_w=np.array(imgBOX_w)
        imgBOX_w=imgBOX_w.reshape(56,112,1)             
       
        
       
       
       
       
        
        
        imgE = pd.read_csv('../CA/data_all/map5/' +img_name, header=None)
        imgE=np.array(imgE)
        #imgD=imgD.reshape(56,112,1)
        locpx,locpy=np.where(imgE==1)
        
        #print(locpx.shape)
        imgE=imgE.reshape(56,112,1)
        
        
       

        #args = (Z_image,imgA, imgB,imgC, imgD,imgE, imgF)
        #args = (imgZ,imgA, imgB,imgC, imgD,imgE, imgF)
        args = (imgA,imgC,imgD)
        #args = (imgA, imgB)
        #args = (imgG,imgH,imgI)
        img = np.concatenate(args,axis=2)
        
        
        #print(img.shape)
        img = img.transpose(2,0,1)
        img = torch.FloatTensor(img)


       





        imgtarget = pd.read_csv('../CA/data_all/maptarget/' +img_name, header=None)
        imgtarget=np.array(imgtarget)
        locx,locy=np.where(imgtarget==1)
        if len(locx)>1:
            locx,locy=np.where(imgtarget==0)
            
        imgtarget=imgtarget.reshape(56,112,1)
        imgtarget0=np.ones((56,112,1))-imgtarget
        #imgtarget = np.concatenate((imgtarget,imgtarget0),axis=2)
        
       
        imgtarget = imgtarget.transpose(2,0,1)
        imgtarget = torch.FloatTensor(imgtarget)
        
        shuttle_vx = pd.read_csv('../CA/data_all/map11/' +img_name, header=None)        
        shuttle_vx=np.array(shuttle_vx)   
        shuttle_vx=shuttle_vx.reshape(56,112,1)
        
        
              
        shuttle_vy = pd.read_csv('../CA/data_all/map12/' +img_name, header=None)
        shuttle_vy=np.array(shuttle_vy)
        shuttle_vy=shuttle_vy.reshape(56,112,1)

        keypoints1= pd.read_csv('../CA/data_all/keypoints1/' +img_name, header=None)
        keypoints1=np.array(keypoints1).reshape(1,-1)
        
        keypoints2= pd.read_csv('../CA/data_all/keypoints2/' +img_name, header=None)
        keypoints2=np.array(keypoints2).reshape(1,-1)
        
        
        
        

      
        return Rx,Ry,Sx,Sy,keypoints1,keypoints2, img,imgtarget ,img_name,locx,locy,locpx,locpy,loc4px,loc4py,imgC,imgD,shuttle_vx,shuttle_vy







