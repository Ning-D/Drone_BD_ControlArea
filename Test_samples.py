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
from BagDataset.Test_noncontrol import BagDataset1
#from net import VGGNet
#from net import FCN8s
cudnn.benchmark = True
plt.ion()   # interactive mode
from tqdm import tqdm
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
import torch.optim as optim
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
from PIL import Image
import visdom

from fcn import VGGNet, FCN8s, FCNs,UNet,NestedUNet
import matplotlib.patches as patches
# 下面开始训练网络
np.set_printoptions(threshold=np.inf)
import seaborn as sns
import sys
from configuration import lr,momentum,w_decay,step_size,gamma,epo_num , batch_size, ratio,ratio1,ALPHA,BETA,GAMMA  
from scipy.ndimage import gaussian_filter
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

batch_size=1

import argparse
parser = argparse.ArgumentParser()
np.set_printoptions(threshold=np.inf)
parser.add_argument('--checkpoint_path', type=str, default='1.pth',
                    help='checkpoint path')

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith('.pth'):
    print("Not a valid checkpoint path! Please modify path in parser.py --checkpoint_path")
    sys.exit(1)


# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

 # 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu






def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated



W = 112  # width of heatmap
H = 56  # height of heatmap
SCALE = 8 # increase scale to make larger gaussians

height = 2160
width = 3840

#width 640
#height 360


def test(show_vgg_params=False):


    adj = torch.tensor([
                         [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 
                         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                         ],
                         dtype=float)





    loss_01=0
    loss_02=0
    loss_03=0
    loss_04=0
    loss_05=0
    loss_06=0
    loss_07=0
    loss_08=0
    loss_09=0
    loss_010=0
 
    #实例化数据集
    #vis = visdom.Visdom()
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])



    
    bag = BagDataset1(transform)
   
    #train_size = int(ratio * len(bag))
    test_size = len(bag)
    print(len(bag)) 
    #test_dataset = random_split(bag, [train_size, test_size])
    test_dataset=bag
    #利用DataLoader生成一个分batch获取数据的可迭代对象
    #train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    #fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)    
    #fcn_model = FCNs(pretrained_net=vgg_model, n_class=1)
    
    
    fcn_model = UNet(num_classes=1, input_channels=27)
    
    fcn_model = fcn_model.to(device)
    fcn_model.load_state_dict(torch.load(str(checkpoint_path)))
    
    
    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = nn.L1Loss()  
    #criterion = nn.BCELoss()
    # 随机梯度下降优化，学习率0.001，惯性分数0.7
    #optimizer = optim.SGD(fcn_model.parameters(), lr=1e-4, momentum=0.7)




    optimizer = optim.Adam(fcn_model.parameters(), lr=lr,weight_decay=w_decay)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
 
    # 记录训练过程相关指标
    all_train_loss = []
    all_test_loss = []
    test_Acc = []
    test_mIou = []
    # start timing
    prev_time = datetime.now()
    eps=[]
    All_control_area_T1=[]
    All_control_area_T2=[]
    control_area_T1=0
    control_area_T2=0
    #for epo in tqdm(range(epo_num)):
    previous_game_name='MD12' 
    previous_rally_name='0'
    ['MD12','MD13','MD14','MD15','MD16','MD17']
        # 验证
    test_loss = 0
    fcn_model.eval()
    with torch.no_grad():
        for index, ( Rx,Ry,Sx,Sy,keypoints1,keypoints2,bag, bag_msk,name,locx,locy,locpx,locpy,loc4px,loc4py,imgC,imgD,shuttle_vx,shuttle_vy) in enumerate(test_dataloader):
                #imshow(bag_msk)
            b=bag
            px=locpx
            #print(px.shape)
            bb=bag_msk
            
            py=locpy
            
            locsx=locx
            locsy=locy
            M=locy
           
            adj=adj.to(device)
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            keypoints1 = keypoints1.to(device)
            keypoints2 = keypoints2.to(device)
            Rx=Rx.to(device)
            Ry=Ry.to(device)
            Sx=Sx.to(device)
            Sy=Sy.to(device)
            optimizer.zero_grad()
            output = fcn_model(bag.float(),keypoints1.float(),keypoints2.float(),adj.float(),Rx,Ry,Sx,Sy)
            
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            #output = torch.softmax(output,dim=1) # output.shape is torch.Size([4, 2, 160, 160])           
            loss_batch=0
            for j in range(len(bag_msk)): 
                    #print(bag_msk[j].shape)
                    #print(bag_msk[j,:,locx[j],locy[j]])   
                    output[output < 0.5] = 0           
                    loss = criterion(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])
                    keyword = 'Flip'
                    if keyword not in name[j]:
                        
                        if 0<=loss<0.1:
                            loss_01+=1
                        if 0.1<=loss<0.2:
                            loss_02+=1
                        if 0.2<=loss<0.3:
                            loss_03+=1
                        if 0.3<=loss<0.4:
                            loss_04+=1
                        if 0.4<=loss<0.5:
                            loss_05+=1
                        if 0.5<=loss<0.6:
                            loss_06+=1
                        if 0.6<=loss<0.7:
                            loss_07+=1
                        if 0.7<=loss<0.8:
                            loss_08+=1
                        if 0.8<=loss<0.9:
                            loss_09+=1
                        if 0.9<=loss<1:
                            loss_010+=1                                              
                        
                        print('%s, loss = %f' %(name, loss))
                    loss_batch=loss+loss_batch
            #print(output[:,:,locx,locy].shape)
            #loss = criterion(output*bag_msk,bag_msk)
                
            loss=loss_batch/float(batch_size)
            iter_loss = loss.item()
                
            test_loss += iter_loss
            output_np = output.cpu().detach().numpy().copy() 
           
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() 
            px = px.cpu().detach().numpy().copy() 
            py = py.cpu().detach().numpy().copy() 
            locsx = locsx.cpu().detach().numpy().copy() 
            locsy = locsy.cpu().detach().numpy().copy() 
            loc4px=loc4px.cpu().detach().numpy().copy() 
            loc4py=loc4py.cpu().detach().numpy().copy() 
            M=M.cpu().detach().numpy().copy() 
           
   
            keyword = 'Flip'
            for j in range(len(bag)):
               
               im=output_np[j][0].reshape(H,W)
               #print(im.shape)
               #print(a)  
               name=np.array(name)
               
               #print(name)
               n=name[j].replace('.csv','.png')
               
               if keyword in n:
                   #print(n)
                   break
               #print('NNNNNNNNNNNN',n)
               fig, axes = plt.subplots(nrows=1, ncols=2,
                               sharex=True, sharey=False,
                               figsize=(19, 8))
               ax1, ax2 = axes                
               #ax1.matshow(im)                
               #f, ax = plt.subplots(figsize=(16, 9))
               #sns.heatmap(im, cmap='Reds',vmax=1, vmin=0, center=0.5,square=True,annot=True)
               #sns.heatmap(bb[j][0]*255, cmap='Reds',vmax=1, vmin=0, center=0.5,square=True,annot=True)
               #print(bb[j][0].max())
               im1=im
               if M[j]>int(W/2):
                   im[:,:int(W/2)]=0

               else:
                   im[:,int(W/2):]=0
           
               p1,p2=np.where(im>=0.5)
               #print(len(p1))
              
               game_name=os.path.splitext(n)[0].split('_')[0]
               rally_name=os.path.splitext(n)[0].split('_')[1]
              # print('game_name',game_name)
               if game_name==previous_game_name and rally_name==previous_rally_name:
                  
                   #print('X',loc4px[0][0])
                   #print('Y',loc4py[0][0])
                   #print(loc4py[0][0])
                   #print(M[j])
                   
                   if (loc4px[0][0]-width/2)* (M[j]-int(W/2))>0:
                   
                       control_area_T1=control_area_T1+len(p1)
                       #print('T1-Name = %s, %d'
                #%(n, len(p1))) 
                   else:
                       control_area_T2=control_area_T2+len(p1)
                       #print('T2-Name = %s, %d'
                #%(n, len(p1))) 
                       
                       
                       
               else:
                  
                 #  print('control_area_T1',control_area_T1)
                 #  print('control_area_T2',control_area_T2)

                   All_control_area_T1.append(control_area_T1)
                   All_control_area_T2.append(control_area_T2)
                   control_area_T1=0
                   control_area_T2=0
                #    if (loc4px[0][0]-width/2)* (M[j]-int(W/2))>0:
                #
                #        control_area_T1=len(p1)
                #
                #       # print('T1-Name = %s, %d'
                # #%(n, len(p1)))
                #    else:
                #        control_area_T2=len(p1)
                 #      print('T2-Name = %s, %d'
                #%(n, len(p1)))                   
                        
                   
                   
               previous_game_name=game_name
               previous_rally_name=rally_name
               
               #sns.heatmap(im, cmap='RdBu',square=True,annot=False,ax=ax1,cbar_kws={"shrink": 0.641})
              
               sns.heatmap(im, cmap='RdBu',square=True,annot=False,ax=ax1,cbar_kws={"shrink": 0.641},vmax=1, vmin=0, center=0.5)
                              
              # locxs,locys=np.where(px[j][0]==1)
               #px=px.view(-1, 2)
               #print(px)
              # print(px.shape[1])
               imgC=imgC.reshape(-1,H,W)
               imgD=imgD.reshape(-1,H,W)
               #print(imgC.shape)
               #print(px)
               for c in range(2):
               #if True
                  ax1.scatter(py[j][c], px[j][c], s=100,facecolors='none', linewidths=2, edgecolors="orange")
                  ax1.quiver(py[j][c], px[j][c],imgC[j,px[j][c],py[j][c]],imgD[j,px[j][c], py[j][c]],angles='xy', scale_units='xy', scale=1)
                  if py[j][c]==0 and px[j][c]==0:
                      print(str(n))
                      print(a)
                  #ax1.scatter(py[j][c], px[j][c], s=100, alpha=0.5, linewidths=2,edgecolors="orange")
               #locsx,locsy=np.where(b[j][0]==1)
               
               #print(x)
               #print(a)
               ax1.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
               ax1.annotate((locsy[j][0],locsx[j][0]), xy = (locsy[j], locsx[j]), size = 15, color = "red")
               ax1.vlines(x=170*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=3670*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=1920*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1,linestyle = "dashed")
               ax1.vlines(x=370*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=3470*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=1395*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=2445*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
      
               ax1.hlines(y=280*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1880*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=405*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1755*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1080*H/height, xmin=170*W/width,xmax=1395*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1080*H/height, xmin=2445*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               
               
               #plt.show()

              
      
               
               
               smoothed_matrix = gaussian_filter(im1, sigma=2,mode='nearest')
               if M[j]>int(W/2):
                   smoothed_matrix[:,:int(W/2)]=0
               else:
                   smoothed_matrix[:,int(W/2):]=0                       
               
               
               
               #ax2.matshow(smoothed_matrix)
               #sns.heatmap(smoothed_matrix, cmap='RdBu',square=True,annot=False,ax=ax2,cbar_kws={"shrink": 0.641})
               sns.heatmap(smoothed_matrix, cmap='RdBu',square=True,annot=False,ax=ax2,cbar_kws={"shrink": 0.641},vmax=1, vmin=0, center=0.5)
               
               for c in range(2):
               
                  ax2.scatter(py[j][c], px[j][c], s=100,facecolors='none', linewidths=2, edgecolors="orange")
                  #ax2.quiver(py[j][c], px[j][c],imgD[j,px[j][c],py[j][c]],imgC[j,px[j][c], py[j][c]])
                  ax2.quiver(py[j][c], px[j][c],imgC[j,px[j][c],py[j][c]],imgD[j,px[j][c], py[j][c]],angles='xy', scale_units='xy', scale=1)
                  #ax2.quiver(py[j][c], px[j][c],vx[j][c],vy[j][c])
                  #ax1.scatter(py[j][c], px[j][c], s=100, alpha=0.5, linewidths=2,edgecolors="orange")
               #locsx,locsy=np.where(b[j][0]==1)
               
               #print(x)
               #print(a)
               ax2.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
                   
                   
               ax2.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
               #ax2.annotate((locsy[j][0],locsx[j][0]), xy = (locsy[j], locsx[j]), size = 15, color = "red")
               
               
               
               ax2.vlines(x=170*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=3670*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=1920*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1,linestyle = "dashed")
               ax2.vlines(x=370*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=3470*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=1395*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=2445*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
      
               ax2.hlines(y=280*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1880*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=405*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1755*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1080*H/height, xmin=170*W/width,xmax=1395*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1080*H/height, xmin=2445*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               
              
               
               
               
               
               
               
               
               
               
               c_path=checkpoint_path.replace('.','')
               c_path=c_path.replace('pth','')              
               
               
               
               
               path = './Performance/'+c_path
               isExist = os.path.exists(path)
               if not isExist:
                   os.makedirs(path)
               plt.ioff()
               
               
               fig.tight_layout()
               plt.savefig(str(path)+'/'+str(n))  
               plt.close(fig)
              
   
        all_test_loss.append(test_loss/len(test_dataloader))


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        
        #print('<---------------------------------------------------->')
        #print('epoch: %f'%epo)
        print('All Test loss= %f, %s'
                %(test_loss/len(test_dataloader), time_str))        
        a=test_loss/len(test_dataloader)
        with open('All test loss'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'.txt', 'w') as f:
            f.write('%f' % a)
       # print(All_control_area_T1)
       # print(All_control_area_T2)


    ii=[]
    loss_pie=[]
    plt.figure(figsize=(7, 7), )
    for i in range(1,11):
        loss_pie.append(locals()["loss_0"+str(i)])
        ii.append(i)
    label = ["0~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4", "0.4~0.5","0.5~0.6", "0.6~0.7", "0.7~0.8", "0.8~0.9", "0.9~1.0"]
    x = np.array(loss_pie)
    plt.tight_layout()
    plt.pie(x,labels=label)
    
    plt.savefig('All test loss'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_pie.png')
    plt.clf()
    y_pos=x
    plt.bar(ii, x, align="center")
    
    bars = ["0~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4", "0.4~0.5","0.5~0.6", "0.6~0.7", "0.7~0.8", "0.8~0.9", "0.9~1.0"]
    plt.xticks(ii, bars, rotation=45)
    plt.savefig('All test loss'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_bar.png')
    print(loss_01)
    print(loss_02)
    print(loss_03)           
    print(loss_04)
    print(loss_05)
    print(loss_06) 
    print(loss_07)
    print(loss_08)
    print(loss_09) 
    print(loss_010)















if __name__ == "__main__":

    test(show_vgg_params=False)





