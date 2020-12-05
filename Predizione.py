import os
os.chdir(r'C:\Users\Carniani\Desktop\Project')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import scipy as sc
from sklearn import metrics
from sklearn import model_selection 
from sklearn import feature_selection 
from sklearn.pipeline import Pipeline
from CaricaDati import CaricaPickles
from CreaDB2 import DBWhole
from OpenRaw import OpenRaw
from Resampling import Resampling
from MessaInDepth import MessaInDepth
from joblib import dump, load
import matplotlib.gridspec as grds
from Metrics import SignalStats
# This program allow to perform automatic depth matching by loading a dataset and performing a 2-steps-Shift using 2 neural networks. The probability of shift are visualized
# in a subplot
listaMov=[1.5,-1.25,9.5,0.4,0.25,4] #This are the shift value sampled using the Depth matching program
#Loading test dataset
N=90
[Cores2,Logs2]=OpenRaw(2,0) 
[RicCores2,RicLogs]=Resampling(Logs2,Cores2,N,0) 
BLIND=DBWhole(RicCores2,RicLogs,N,listaMov,10) 
Xt=BLIND[:,0:-1]
yt=BLIND[:,-1]

#Loading models
BestPipe=load('./MLModels/MigliorMod90BT2Final')# NN for bulk shift
BestPiperef=load('./MLModels/MigliorMod90BT2Final') # NN for refined shift

# Prediction of the original and transformed signal
yp2=BestPipe.predict_proba(Xt)
Div=int((BLIND.shape[1]-1)/2-1)
Xt3=-Xt
yprib=BestPipe.predict_proba(Xt3)
Xt4=np.concatenate([np.flip(Xt[:,0:Div+1],axis=1),np.flip(Xt[:,Div+1:Div*2+2],axis=1)],axis=1)
ypflip=BestPipe.predict_proba(Xt4)
Xt5=-Xt4
ypribflip=BestPipe.predict_proba(Xt5)
ypnoise=BestPipe.predict_proba(Xt+np.random.normal(0,0.3,Xt.shape))
yp3=(yp2[:,1] +ypribflip[:,1]+yprib[:,1]+ypflip[:,1]+ypnoise[:,1])*0.2 #Averaging probabilities
i=0

# From the indices to the shift values

fromto=[]    
for cores in RicCores2:
     fromto.append(int(10/np.diff(cores.Depth).min())) 
     
for i in range(0,6,1):
     print(i) Sec=yp3[i*Space:(i+1)*Space].copy()
     Sec=Sec/max(Sec)
     Pos.append(np.argmin(abs(Sec-Sec.max())))
     ypz[0,i*Space+Pos[i]]=1

ypz=np.zeros([1,len(BLIND)])
space=0;j=0;k=0
Sec=[];ypz2=[];Pos=[]
fig=plt.figure(figsize=(10,7))
gs= grds.GridSpec(2,3,fig )

Mossa=[]
for i in fromto: # Subplots of probabilities
     D=np.diff(RicCores2[j].Depth).min()
     space1=space
     space=i*2+space
     Sec.append(yp3[space1:space].copy())
     Sec2=Sec[j]/max(Sec[j])
     Pos.append(np.argmin(abs(Sec2-Sec2.max())))
     ypz[0,space1+Pos[j]]=1   
     ypz2.append(ypz[0,space1:space])
     print(Pos[j]*D-i*D)
     Mossa.append(round(Pos[j]*D-i*D,2))
     fig.add_subplot(gs[int(j/3),k] )
     plt.plot(np.linspace(-(i+1)*D,(i+1)*D,i*2)-listaMov[j],Sec[j])
     plt.plot(np.linspace(-(i+1)*D,(i+1)*D,i*2)-listaMov[j],ypz2[j])
     if k==0:
          plt.ylabel('Match Probability')
     plt.ylim([0,1])
     plt.xlabel('Distance from match (m)')
#     plt.legend()
     plt.tight_layout()
     plt.grid()
     j=j+1
     k=k+1
     if k>2:
          k=0
          
#____________________________________________________________

[SCores2,Logs2]=OpenRaw(2,1) 
N_Spezzoni=[]
for i in SCores2:
    for j in i:
        N_Spezzoni.append(len(j))

[SRicCores2,SRicLogs2]=Resampling(Logs2,SCores2,N,1)    

j=0
Mossa2=[];listamov2=[]
for i in N_Spezzoni:
     Mossa2.append(np.repeat( round(Mossa[j],2),i ))
     listamov2.append(np.repeat( listaMov[j],i ))
     j=j+1

Mossa2=[item for items in Mossa2  for item in items]
listamov2=[item2 for items2 in listamov2  for item2 in items2]
Mossatot=np.array(Mossa2)-np.array(listamov2)
Mossatot2=np.array(Mossa)-np.array(listaMov)

BLIND2=DBWhole(SRicCores2, SRicLogs2,N,Mossatot,3)
 
j=0
Xt=BLIND2[:,0:-1]
yt=BLIND2[:,-1]
yp2=BestPiperef.predict_proba(Xt)
Div=int((BLIND2.shape[1]-1)/2-1)
Xt3=-Xt
yprib=BestPiperef.predict_proba(Xt3)
Xt4=np.concatenate([np.flip(Xt[:,0:Div+1],axis=1),np.flip(Xt[:,Div+1:Div*2+2],axis=1)],axis=1)
ypflip=BestPiperef.predict_proba(Xt4)
Xt5=-Xt4
ypribflip=BestPiperef.predict_proba(Xt5)
ypnoise=BestPiperef.predict_proba(Xt+np.random.normal(0,0.3,Xt.shape))
yp3=(yp2[:,1] +ypribflip[:,1]+yprib[:,1]+ypflip[:,1]+ypnoise[:,1])*0.2
i=0
 
fromto=[]  

for cores in SRicCores2:
     fromto.append(int(3/np.diff(cores.Depth).min()))


ypz=np.zeros([1,len(BLIND2)])
space=0;j=0;k=0;p=0
Sec=[];ypz2=[];Pos=[]
fig=plt.figure(figsize=(10,7))
gs= grds.GridSpec(2,3,fig )

Mossa=[]
for i in fromto:
    D=np.diff(SRicCores2[j].Depth).min()
    space1=space
    space=i*2+space
    Sec.append(yp3[space1:space].copy())
    Sec2=Sec[j]/max(Sec[j])
    Pos.append(np.argmin(abs(Sec2-Sec2.max())))
    Mossa.append(round(Pos[j]*D-i*D,2))  
    j=j+1
    if j==N_Spezzoni[p]:
        Sec=sum(Sec)/N_Spezzoni[p]
        fig.add_subplot(gs[int(p/3),k] )
        plt.plot(np.linspace(-(i+1)*D,(i+1)*D,i*2)-Mossatot2[p],Sec)
        # plt.plot(np.linspace(-(i+1)*D,(i+1)*D,i*2)-Mossatot[j],ypz2[j])
        plt.grid()
        if k==0:
            plt.ylabel('Match Probability')
            plt.ylim([0,1])
            plt.xlabel('Distance from match (m)')
            plt.tight_layout()
        Sec=[]
        k=k+1
        if k>2:
             k=0
        p=p+1        
        j=0
 
