import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
from scipy import signal

def BuildDB(RicC2,RicL2): #Core and log data
    N=90
    MovA=1
    Ind=(N-MovA+1)
    Dim=100
    Ind2=Ind+Dim*2
    DATASET= []
    weights = np.ones(MovA) / MovA
    y=[]
    for j in range(0,len(RicC2),1):
        u=np.argmin(abs(RicC2[j].Depth[0]-RicL2[j].Depth))  
        Dr=100
        w2=np.random.randint(-Dr,Dr,1)
        print('Siamo a '+str(j)+' cores nel dataset' )
        for i in range(-Dr,Dr,1):
            Arr=np.zeros([1,Ind2*2])
            if u -Dr+w2< 0:
                w2=u-Dr
            RLC=np.convolve( RicL2[j].Gr1[int(u+w2-Dr): int(u+w2+Dr+N)] , weights, mode='valid')  # Moving Average
            RLC=(RLC-RLC.mean())/RLC.std() #Normalization
            if len(RLC) < (Ind2):
                for k in range(0,Ind2-len(RLC),1):
                    RLC=np.append(RLC,0)
            Arr[0, 0:len(RLC)]=RLC.copy()  
            RCC=np.convolve(np.array(RicC2[j].Gr1) , weights, mode='valid')
            RCC=(RCC-RCC.mean())/RCC.std()
 
            RLC[i+Dr:Ind+i+Dr]=RCC
            Arr[0, len(RLC):len(RLC)*2 ]=RLC 
            y.append(int(i+w2))
            DATASET.append(Arr)
            if  i+w2>=-1 and i+w2<=1: #If we are in the match position 
#                Pass=Arr.copy()
#                Pass[0,i+Dr:Ind+i+Dr  ]=Pass[0,i+Dr:Ind+i+Dr ]+np.random.normal(0,0.3,N)
#                Pass[0, i+Dr+len(RLC):Ind+i+Dr+len(RLC)]=Pass[0,\
#                     i+Dr+len(RLC):Ind+i+Dr+len(RLC)]+np.random.normal(0,0.3,N)
#                DATASET.append(Pass)
#                y.append(0)
#                DATASET.append(-Pass)
#                y.append(0)
#                DATASET.append(-Arr)
#                y.append(0)
#                Arr[0,0:len(RLC)]=np.flip(Arr[0,0:len(RLC)])
#                Pass[0,0:len(RLC)]=np.flip(Pass[0,0:len(RLC)])
#                Arr[0,len(RLC):len(RLC)*2]=np.flip(Arr[0,len(RLC):len(RLC)*2])
#                Pass[0,len(RLC):len(RLC)*2]=np.flip(Pass[0,len(RLC):len(RLC)*2])
#                y.append(0)
#                DATASET.append(Arr)
#                y.append(0)                
#                DATASET.append(Pass)
#                y.append(0)
#                DATASET.append(-Arr)
#                y.append(0)
#                DATASET.append(-Pass)
#                del Pass,Arr                
#            if  i+w2<-1:
#                y.append(1)
#                DATASET.append(Arr)
#                if (i+w2)%15==0: 
#                     Pass=Arr.copy()
#                     Pass[0,i+Dr:Ind+i+Dr  ]=Pass[0,i+Dr:Ind+i+Dr ]+np.random.normal(0,0.3,N)
#                     Pass[0, i+Dr+len(RLC):Ind+i+Dr+len(RLC)]=Pass[0,\
#                          i+Dr+len(RLC):Ind+i+Dr+len(RLC)]+np.random.normal(0,0.3,N)
#                     
#                     DATASET.append(Pass)
#                     y.append(1)
#                     
#                     DATASET.append(-Arr)
#                     y.append(1)
#                     
#                     DATASET.append(-Pass)
#                     y.append(1)
#               
#                     Arr[0,0:len(RLC)]=np.flip(Arr[0,0:len(RLC)])
#                     Pass[0,0:len(RLC)]=np.flip(Pass[0,0:len(RLC)])
#                     Arr[0,len(RLC):len(RLC)*2]=np.flip(Arr[0,len(RLC):len(RLC)*2])
#                     Pass[0,len(RLC):len(RLC)*2]=np.flip(Pass[0,len(RLC):len(RLC)*2])
#                     y.append(2)
#          
#                     DATASET.append(Arr)
#                     y.append(2)                
#                     DATASET.append(Pass)
#                     y.append(2)
#                     DATASET.append(-Arr)
#                     y.append(2)
#                     DATASET.append(-Pass) 
#                     del Arr ,RLC ,RCC, Pass
#            if  i+w2>1:
#                y.append(2)
#                DATASET.append(Arr)
#                if (i+w2)%15==0: 
#                     Pass=Arr.copy()
#                     Pass[0,i+Dr:Ind+i+Dr  ]=Pass[0,i+Dr:Ind+i+Dr ]+np.random.normal(0,0.3,N)
#                     Pass[0, i+Dr+len(RLC):Ind+i+Dr+len(RLC)]=Pass[0,\
#                          i+Dr+len(RLC):Ind+i+Dr+len(RLC)]+np.random.normal(0,0.3,N)
#                     DATASET.append(Pass)
#                     y.append(2)
#                     DATASET.append(-Arr)
#                     y.append(2)
#                     DATASET.append(-Pass)
#                     y.append(2)
#                     Arr[0,0:len(RLC)]=np.flip(Arr[0,0:len(RLC)])
#                     Pass[0,0:len(RLC)]=np.flip(Pass[0,0:len(RLC)])
#                     Arr[0,len(RLC):len(RLC)*2]=np.flip(Arr[0,len(RLC):len(RLC)*2])
#                     Pass[0,len(RLC):len(RLC)*2]=np.flip(Pass[0,len(RLC):len(RLC)*2])
#                     y.append(1)
#                     DATASET.append(Arr)
#                     y.append(1)                
#                     DATASET.append(Pass)
#                     y.append(1)
#                     DATASET.append(-Arr)
#                     y.append(1)
#                     DATASET.append(-Pass)
#                     del Arr ,RLC ,RCC, Pass
            
    DATASET=np.array(DATASET)
    DATASET=np.squeeze(DATASET,axis=1)
    y=np.array(y)
    y.shape=[y.shape[0],1]
    DATASET=np.append(DATASET,y,axis=1)
    return(DATASET)


