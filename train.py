import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch_snippets import *
import h5py
from torch.utils.data import random_split
from torchsummary import summary
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure

device='cuda' if torch.cuda.is_available() else 'cpu'#else mps for mac


tfm=T.Compose([ 
    T.ToTensor()
])

class Preprocess(Dataset):
    def __init__(self,path_A,path_B):
        self.path_A=glob.glob(path_A+'/images/*.jpg')
        self.path_B=glob.glob(path_B+'/images/*.jpg')
        self.ground_maps_A=glob.glob(path_A+'/ground-truth-h5/*.h5')
        self.ground_maps_B=glob.glob(path_B+'/ground-truth-h5/*.h5')
        self.total_images=self.path_A+ self.path_B
        self.total_ground_maps=self.ground_maps_A+self.ground_maps_B
        self.total_images.sort()
        self.total_ground_maps.sort()
 
        
    def __len__(self):
        return len(self.total_images)
    
    def __getitem__(self,ix):
        img=read(self.total_images[ix],1)
        with h5py.File(self.total_ground_maps[ix],'r') as f:
            data=np.array(f['density'])    
        resized_data=resize(data,1/8)*64
        return img.copy(),resized_data.copy()



    def collate_fn(self,batch):
        ims,gts=list(zip(*batch))
        ims=torch.cat([tfm(im)[None] for im in ims]).to(device)
        gts=torch.cat([tfm(gt)[None] for gt in gts]).to(device)
        return ims,gts



def get_data():
    trn_ds=Preprocess('/kaggle/working/ShanghaiTech/part_A/train_data','/kaggle/working/ShanghaiTech/part_B/train_data')
    train_size = int(0.8 * len(trn_ds))
    val_size = len(trn_ds) - train_size
    train_dataset, val_dataset = random_split(trn_ds, [train_size, val_size])
    #test_dl=Preprocess('/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_A/test_data','/kaggle/input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/test_data')
    trn_dl=DataLoader(train_dataset,batch_size=1,shuffle=True,collate_fn=trn_ds.collate_fn)
    val_dl=DataLoader(val_dataset,batch_size=1,collate_fn=trn_ds.collate_fn)
    #test_ds=DataLoader(test_dl,batch_size=1,shuffle=True,collate_fn=test_dl.collate_fn)
    return trn_dl,val_dl

class CSRnet(nn.Module):
    def __init__(self):
        super(CSRnet, self).__init__()
        vgg_16=torchvision.models.vgg16(pretrained=True)
        
        self.frontend=nn.Sequential(
            *list(vgg_16.features.children())[:10],
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(), )


        
        self.backend=nn.Sequential(
            
                     
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1),
        )
        
    def forward(self,x):

        x=self.frontend(x)
        x=self.backend(x)

        return x
    
# model=CSRnet()
# summary(model,(3,600,800))


def get_model():
    model=CSRnet().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-6)
    loss_fn=nn.MSELoss()
    return model,loss_fn,optimizer


def train_batch(data,model,loss_fn,optimizer):
    model.train()
    optimizer.zero_grad()
    x,y=data
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    psnr = PeakSignalNoiseRatio().to(device)
    ssi=StructuralSimilarityIndexMeasure().to(device)
    noise=psnr(y_pred, y)
    ssm=ssi(y_pred,y)
    pts_loss=nn.L1Loss()(y_pred.sum(),y.sum())
    return loss.item(),pts_loss.item(),noise,ssm

@torch.no_grad()
def validate_batch(data,model,loss_fn):
    model.eval()
    x,y=data
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    psnr = PeakSignalNoiseRatio().to(device)
    ssi=StructuralSimilarityIndexMeasure().to(device)
    noise=psnr(y_pred, y)
    ssm=ssi(y_pred,y)
    pts_loss=nn.L1Loss()(y_pred.sum(),y.sum())
    return loss.item(),pts_loss.item(),noise,ssm


trn_dl,val_dl=get_data()
model,criterion,opt=get_model()
n_epochs=100
log=Report(n_epochs)
for ex in range(n_epochs):
    N=len(trn_dl)
    for ix,data in enumerate(trn_dl):
        loss,pts_loss,noise,ss=train_batch(data,model,criterion,opt)
        log.record(ex+(ix+1)/N,trn_loss=loss,trn_pts=pts_loss,trn_psnr=noise,trn_ssi=ss,end='\r')
    N=len(val_dl)
    for ix,data in list(enumerate(val_dl)):
        loss,pts_loss,noise,ss=validate_batch(data,model,criterion)
        log.record(ex+(ix+1)/N,val_loss=loss,val_pts=pts_loss,val_psnr=noise,val_ssi=ss,end='\r')
    
    log.report_avgs(ex+1)

log.plot_epochs()     
    
    
torch.save(model.state_dict(),'model.pth')