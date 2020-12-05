import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc  
from sklearn.pipeline import Pipeline
# This program allows the operator to bring a core in the depth matching position by visualizing the core and the log and writing the number of meters for the shift
def MessaInDepth(RicCore, RicLogs): 
    N=RicCores[0].shape[0]
    R=len(RicCores)
    j=0
    for CORE in RicCore:
         for RicCores in CORE:
            Sign='No'
            while Sign !='yes':
                u=np.argmin(abs(RicCores[j].Depth[0]-RicLogs[j].Depth))
                plt.figure()
                plt.plot(np.asarray(RicLogs[j].Gr1[u:N+u])/max(RicLogs[j].Gr1[u:N+u]),np.asarray(RicLogs[j].Depth[u:N+u]))
                plt.plot(RicCores[j].Gr1/max(RicCores[j].Gr1),RicCores[j].Depth )  
                plt.yticks(np.arange(min(np.asarray(RicLogs[j].Depth[u:N+u])), max(np.asarray(RicLogs[j].Depth[u:N+u]))+1, 1))
                plt.xlabel('Gamma Ray(API)')
                plt.ylabel('Depth (m)')
                plt.grid(axis='y')
                plt.tight_layout()
                plt.show(block=False)
                j=j+1
                print('Di quanto sposto?')
                M=float(input())
                if M == 0: 
                break
                RicCores[j].Depth=RicCores[j].Depth-M
                u=np.argmin(abs(RicCores[j].Depth[0]-RicLogs[j].Depth))
                plt.figure()
                plt.plot(np.asarray(RicLogs[j].Gr1[u:N+u]),np.asarray(RicLogs[j].Depth[u:N+u]))
                plt.plot( RicCores[j].Gr1,RicCores[j].Depth,label='Win core' +str(j) )  
                plt.yticks(np.arange(min(np.asarray(RicLogs[j].Depth[u:N+u])), max(np.asarray(RicLogs[j].Depth[u:N+u]))+1, 1))
                plt.xlabel('Gamma Ray(API)')
                plt.ylabel('Depth (m)')
                plt.grid(axis='y')
                plt.tight_layout()

                plt.show(block=False)
                print('Are you satisfied?')
                Sign=input() #Write yes if you want to break
        RicCores[j].to_pickle('./Messainprof/Cores/Core'+str(j))
        RicLogs[j].to_pickle('./Messainprof/Logs/Log'+str(j))
