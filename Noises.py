import numpy as np
#This program add to a certain signal a gaussian. The porpouse is to simulate a peak of noise.
def G_R_Peaks(N):
     x=np.arange(-5,5,10/N)
     mu=np.random.uniform(-4,4)
     sigma=np.random.uniform(0.1,0.2)
     a=(2.*np.pi*sigma**2.)**-.5 * np.exp(-.5*(x-mu)**2./sigma**2.)
     np.random.randint(-1,1)
     Sign= 1 if np.random.uniform(1,-1)>0 else -1
     a=a*Sign
     return(a)
