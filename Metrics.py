import scipy.stats as stats
import numpy as np
from scipy.signal import hilbert
def SignalStats(A,B):
     samples=50
#     Correlation= stats.pearsonr(A,B)[0]
#     Distance=np.sum((A-B)**2)
#     Window_Corr2=[]
#     for i in range(0,int(len(A)/samples),1):
#          Window_Corr=[]
#          for j in range(-8,8,1):
#               Window_Corr.append(stats.pearsonr(A[i*samples+10:(i+1)*samples+10],B[i*samples+j+10:(i+1)*samples+j+10])[0])
#          Window_Corr2.append(np.sum(Window_Corr))
#     WCor=np.sum(Window_Corr2)
     a1=np.angle(hilbert(A))
     a2=np.angle(hilbert(B))
     sinchrony=1-np.sin(np.abs(a1-a2)/2)
     S=np.mean(sinchrony)
     return(np.array([S]))

     
     
     
     

