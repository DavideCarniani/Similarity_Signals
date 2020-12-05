import os
os.chdir(r'C:\Users\Carniani\Desktop\Project')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import  MLPRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn import model_selection 
from sklearn import feature_selection 
from sklearn import preprocessing
import random
import time
from sklearn.pipeline import Pipeline
from CaricaDati import CaricaPickles
from CreaDB import BuildDB
from CreaDB2 import BuildDB2
from CreaDB2 import BuildDB3
from OpenRaw import OpenRaw
from Resampling import Resampling
from MessaInDepth import MessaInDepth
from joblib import dump, load

##RESAMPLING + DATABASE
DATABASE=pd.read_pickle('./Dato_Finale/Tr90_SWDEPTH')
DATABASE=np.array(DATABASE)
BLIND=pd.read_pickle('./Dato_Finale/Te90_SWDEPTH')
BLIND=np.array(BLIND)
 
Div=int((DATABASE.shape[1]-1)/2-1)

X=DATABASE[:,0:-2]
Xt=BLIND[:,0:-1]
y=DATABASE[:,-1]
#yt=BLIND[:,-1]

cv =model_selection.KFold(n_splits=10,random_state=1)
scale=StandardScaler() 
scale2=MinMaxScaler() 

estimator =  [ ('scale',scale),('clf',MLPRegressor(max_iter=200, verbose=False))]
estimator2 = [ ('scale',scale2),('clf',KNeighborsRegressor())] 
estimator3 = [ ('clf',RandomForestRegressor(random_state=1))]

pipe  = Pipeline(estimator)
pipe2  = Pipeline(estimator2)
pipe3 = Pipeline(estimator3)

param_grid = dict(clf__alpha=np.logspace(-9,-4,6),\
                  clf__hidden_layer_sizes=[(1500,1500),(2000,2000),(2500,2500)]  ) 
param_grid2 = dict(clf__n_neighbors=[i for i in range(1,30,1)],clf__p= [1,2]  )
param_grid3 = dict(clf__n_estimators=[ 1000],clf__max_depth=[20,40, None],\
                   clf__min_samples_leaf=[1,8])

grid = model_selection.GridSearchCV(pipe, param_grid=param_grid,cv=cv,n_jobs=-1,verbose=1,scoring='neg_mean_squared_error')
grid2 = model_selection.GridSearchCV(pipe2, param_grid=param_grid2,cv=cv,n_jobs=-1,verbose=1,scoring='neg_mean_squared_error')
grid3= model_selection.GridSearchCV(pipe3, param_grid=param_grid3,cv=cv,n_jobs=-1,verbose=1,scoring='neg_mean_absolute_error')

models = []
models.append(('Neural Net', grid))
models.append(('K Nearest', grid2))
models.append(('Random Forest', grid3))

results = []
names = []
best_p=[]
results2=[]
Yp=[]
BestEstimator=[]
AveRank=[]

for name, model in models:
   model.fit(X,y) 
   best_p.append(name)
   best_p.append(model.best_params_)
   rescv=model.cv_results_['mean_test_score'][model.best_index_]
   stdcv=model.cv_results_['std_test_score'][model.best_index_]
   BestEstimator.append((name,model.best_estimator_))
   names.append(name)   
   msg = "%s: %f (%f)" % (name,rescv  , stdcv  )
   print(msg)
   AveRank.append((rescv, stdcv,  name))
   
# Confusion Matrix of cross validation  

with open("Log_reg.txt", "w") as text_file:
     print(f"Best results: {best_p}", file=text_file)  
with open("Results_reg.txt", "w") as text_file:
     print(f"Best Scores: {AveRank}", file=text_file)

#BEST ALGORITHMS AND EVALUATION
AveRank.sort(reverse=True)
j=0
for name, pipe in BestEstimator: 
    if AveRank[0][2] is name:
        BestPipe=BestEstimator[j][1]
    j+=1

BestPipe.fit(X,y)
yp=BestPipe.predict(Xt)
 
with open("FinalReport_Reg.txt", "w") as text_file:
    print(f" {yp}", file=text_file)
    
    
    
        
dump(BestPipe,'./MLModels/ConDEPTH')
#BestPipe2=load('./MLModels/MigliorMod_regP')
 
