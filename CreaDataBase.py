import numpy as np
from Noises import G_R_Peaks
from Metrics import SignalStats
from numpy import random
import scipy as sc
from scipy.fftpack import fftshift
import scipy.stats as stats

# RicC2: resampled cores; RicL2: resampled logs
# mov: True shifts; Range(m): researching the match from +/- Range from starting position 

def DBWhole(RicC2,RicL2,N,mov,Range ):
    Ind=(N-1+1)
    DATASET= []
    y=[]
    for j in range(0,len(RicC2),1):
        u=np.argmin(abs((RicC2[j].Depth[0]-mov[j])-RicL2[j].Depth))  # Loading starting position
        Dr= int(Range/np.diff(RicC2[j].Depth).min())
        for i in range(-Dr,Dr,1):
            Arr=np.zeros([1,Ind*2])
            if u +i < 0:
                 i=-u
            RLC=np.array(RicL2[j].Gr1[int(u+i): int(u+i+N)])  
            RLC=(RLC-RLC.mean())/RLC.std()       
            if len(RLC) < (Ind): # If the log is not long enough, zero padding
                for k in range(0,Ind-len(RLC),1):
                    RLC=np.append(RLC,0)
            Arr[0, 0:Ind]=RLC.copy()  #First part of the database's row
            RCC=np.array(RicC2[j].Gr1)   
            RCC=(RCC-RCC.mean())/RCC.std()   #Second part of the database's row
            Arr[0, Ind:Ind*2]=RCC.copy()
            if i>=-1 and i<=1:  #Relaxing prediction by including next to the match points
                y.append(1)
                DATASET.append(Arr)
            if i != 0 and i!= -1 and i!=1 :
                y.append(0)
                DATASET.append(Arr)
    DATASET=np.array(DATASET)
    DATASET=np.squeeze(DATASET,axis=1)
    y=np.array(y)
    y.shape=[y.shape[0],1]
    DATASET=np.append(DATASET,y,axis=1)   
    return(DATASET)
  
    
