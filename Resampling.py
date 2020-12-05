import pandas as pd
import numpy as np
from scipy import signal

def Resampling(Logs,Cores,N,Flag):   
    RicCores=[]
    RicLogs=[] 
    j=0
    if Flag==0:
         for C in Cores:
             for c in C:
     #             for c in z: 
                  B=np.arange(min(c.Depth),max(c.Depth),min(np.diff(c.Depth)))
                  A=np.interp(B,c.Depth,c.Gr1)
                  [Gr,De]=signal.resample(A, N, t=B)
                  Data={'Depth': De, 'Gr1': Gr}
                  RicCores.append(pd.DataFrame(Data))
                  dDc=min(np.diff(De))
                  dDl=min(np.diff(Logs[j].Depth))
                  Length=Logs[j].shape[0]
                  [Grl,Del]=signal.resample(np.array(Logs[j].Gr1), int(Length*(1/(dDc/dDl))) , np.array(Logs[j].Depth))
                  DataL={'Depth': Del, 'Gr1':Grl }
                  RicLogs.append(pd.DataFrame(DataL))
             j=j+1
    else: 
           for C in Cores:
             for z in C:
                  for c in z: 
                       B=np.arange(min(c.Depth),max(c.Depth),min(np.diff(c.Depth)))
                       A=np.interp(B,c.Depth,c.Gr1)
                       [Gr,De]=signal.resample(A, N, t=B)
                       Data={'Depth': De, 'Gr1': Gr}
                       RicCores.append(pd.DataFrame(Data))
                       dDc=min(np.diff(De))
                       dDl=min(np.diff(Logs[j].Depth))
                       Length=Logs[j].shape[0]
                       [Grl,Del]=signal.resample(np.array(Logs[j].Gr1), int(Length*(1/(dDc/dDl))) , np.array(Logs[j].Depth))
                       DataL={'Depth': Del, 'Gr1':Grl }
                       RicLogs.append(pd.DataFrame(DataL))
             j=j+1
        
        
    return(RicCores,RicLogs)

