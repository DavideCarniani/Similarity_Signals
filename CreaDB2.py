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
  
    

#def BuildDB4(RicC2,RicL2):
#    N=90
#    MovA=3
#    Ind=(N-MovA+1)
#    DATASET= []
#    weights = np.ones(MovA) / MovA
#    y=[]
#    rand=np.random.randint(-50,50,1)
#    l=N*3-2
#    for j in range(0,len(RicC2),1):
#        u=np.argmin(abs(RicC2[j].Depth[0]-RicL2[j].Depth))  
#        print('Siamo a '+str(j)+' cores nel dataset' )
#        Arr=np.zeros([1,l*2+ Ind+1])
#        RLC=np.convolve( RicL2[j].Gr1[int(u-rand-3/2*N): int(u-rand+N*3/2)] , weights, mode='valid')  
#        Depth=RicL2[j].Depth[int(1+u-rand-3/2*N): int(u-rand+N*3/2-1)] 
#        RLC=(RLC-RLC.mean())/RLC.std()
#        if len(RLC) < (l):
#             RLC=np.append(RLC,0)
#             for k in range(0,l-len(RLC),1):
#                  RLC=np.append(RLC,0)
#                  Depth=np.append(Depth,0)
#
#        Arr[0, 0:l]=RLC.copy()  
#        RCC=np.convolve(np.array(RicC2[j].Gr1) , weights, mode='valid')
#        RCC=(RCC-RCC.mean())/RCC.std()
#        Arr[0, l:l+Ind]=RCC.copy()
#        Arr[0, l+Ind:Ind+l*2]=Depth.copy()
#        y.append(RicC2[j].Depth[0])
#        DATASET.append(Arr)
#        
#    DATASET=np.array(DATASET)
#    DATASET=np.squeeze(DATASET,axis=1)
#    y=np.array(y)
#    y.shape=[y.shape[0],1]
#    DATASET=np.append(DATASET,y,axis=1)
#    return(DATASET)
#    
#def BuildDB5(RicC2,RicL2):
#    N=90
#    MovA=1
#    Ind=(N-MovA+1)
#    DATASET= []
#    weights = np.ones(MovA) / MovA
#    y=[]
#    for j in range(0,len(RicC2),1):
#        u=np.argmin(abs(RicC2[j].Depth[0]-RicL2[j].Depth))  
#        Dr= 100
#        print('Siamo a '+str(j)+' cores nel dataset' )
#        for i in range(-Dr,Dr,1):
#            Arr=np.zeros([1,Ind*2])
#            if u +i < 0:
#                 i=-u
#            RLC=np.convolve( RicL2[j].Gr1[int(u+i): int(u+i+N)] , weights, mode='valid')  
#            RLC=(RLC-RLC.mean())/RLC.std()
#            if len(RLC) < (Ind):
#                for k in range(0,Ind-len(RLC),1):
#                    RLC=np.append(RLC,0)
#            Arr[0, 0:Ind]=RLC.copy()  
#            RCC=np.convolve(np.array(RicC2[j].Gr1) , weights, mode='valid')
#            RCC=(RCC-RCC.mean())/RCC.std()
#            Arr[0, Ind:Ind*2]=RCC.copy()
#            if  i>=-1 and i<=1:
#                y.append(1)
#                DATASET.append(Arr) 
#            if i != 0 and i!= -1 and i!=1:
#                y.append(0)
#                DATASET.append(Arr) 
#
#    DATASET=np.array(DATASET)
#    DATASET=np.squeeze(DATASET,axis=1)
#    y=np.array(y)
#    y.shape=[y.shape[0],1]
#    DATASET=np.append(DATASET,y,axis=1)
#    return(DATASET)      
#    
#    
#    
# def BuildDB2(RicC2,RicL2,N,mov,Range,MovA):
#     Ind=(N-1+1)
#     DATASET= []
#     weights = np.ones(MovA) / MovA
#     y=[]
#     for j in range(0,len(RicC2),1):
        
#         u=np.argmin(abs((RicC2[j].Depth[0]+mov[j])-RicL2[j].Depth))  
#         Dr= int(Range/np.diff(RicC2[j].Depth).min())
        
#         for i in range(-Dr,Dr,1):
            
#             Arr=np.zeros([1,Ind*2])
            
#             if u +i < 0:
#                  i=-u
#             RLC=np.array(RicL2[j].Gr1[int(u+i): int(u+i+N)])  
#             RLC=(RLC-RLC.mean())/RLC.std()       
            
#             if len(RLC) < (Ind): # If the log is not long enough, zero padding
#                 for k in range(0,Ind-len(RLC),1):
#                     RLC=np.append(RLC,0)
            
#             Arr[0, 0:Ind]=RLC.copy()  #First part of the database's row
            
#             if MovA==1: # If it is selected, weigthed average is applied
#                  RCC=np.array(RicC2[j].Gr1)  
#             else:
#                  RCC=sc.ifft(sc.fft(np.array(RicC2[j].Gr1))*\
#                        sc.fft(weights,n=len(RLC))) 
                     
#             RCC=(RCC-RCC.mean())/RCC.std()   #Second part of the database's row
#             Arr[0, Ind:Ind*2]=RCC.copy()
            
#             if i>=-1 and i<=1:        
#                 y.append(1)
#                 DATASET.append(Arr)
# #                y.append(1)
# #                Pass=Arr+np.random.normal(0,0.5,Ind*2)
# #                DATASET.append(Pass) 
# #                y.append(1)
# #                DATASET.append(-Arr)
# #                y.append(1)
# #                DATASET.append(-Pass) 
# #                y.append(1)
# #                Arr2=Arr.copy()
# #                Arr2[0,0:Ind]=np.flip(Arr[0,0:Ind])
# #                Arr2[0,Ind:2*Ind]=np.flip(Arr[0,Ind:2*Ind])
# #                DATASET.append(Arr2)
# #                y.append(1)
# #                Pass2=Arr2+np.random.normal(0,0.5,Ind*2)
# #                DATASET.append(Pass2) 
# #                y.append(1)
# #                DATASET.append(-Pass2) 
# #                y.append(1)
# #                DATASET.append(-Arr2)
#             if i != 0 and i!= -1 and i!=1 :
#                 y.append(0)
#                 DATASET.append(Arr)
# #                if i%15==0:
# #                     y.append(0)
# #                     Pass=Arr+np.random.normal(0,0.5,Ind*2)
# #                     DATASET.append(Pass) 
# #                     y.append(0)
# #                     DATASET.append(-Arr)
# #                     y.append(0)
# #                     DATASET.append(-Pass) 
# #                     y.append(0)
# #                     Arr2=Arr.copy()
# #                     Arr2[0,0:Ind]=np.flip(Arr[0,0:Ind])
# #                     Arr2[0,Ind:2*Ind]=np.flip(Arr[0,Ind:2*Ind])
# #                     DATASET.append(Arr2)
# #                     y.append(0)
# #                     Pass2=Arr2+np.random.normal(0,0.5,Ind*2)
# #                     DATASET.append(Pass2) 
# #                     y.append(0)
# #                     DATASET.append(-Pass2) 
# #                     y.append(0)
# #                     DATASET.append(-Arr2)
          
                
#     DATASET=np.array(DATASET)
#     DATASET=np.squeeze(DATASET,axis=1)
#     y=np.array(y)
#     y.shape=[y.shape[0],1]
#     DATASET=np.append(DATASET,y,axis=1)   
#     return(DATASET)