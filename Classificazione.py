import os
os.chdir(r'C:\Users\Carniani\Desktop\Project')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn import model_selection 
from sklearn.pipeline import Pipeline
from CreaDB2 import BuildDB2 
from OpenRaw import OpenRaw
from Resampling import Resampling
from joblib import dump, load
from Metrics import SignalStats
import matplotlib.gridspec as grds
from sklearn.neural_network import MLPClassifier

#RESAMPLING + DATABASE
N=90

#Load Dataset
nome='BT2Finale_Spezzoni'
print(nome)
DATABASE=pd.read_pickle('C:/Users/Carniani/Desktop/Project/Dato_Finale/'+ nome)
DATABASE=np.array(DATABASE)
X=DATABASE[:,0:-1]
y=DATABASE[:,-1]

#Set up pipelines and cross validation
cv =model_selection.KFold(n_splits=5)
scale=StandardScaler() 
scale2=MinMaxScaler() 

estimator =  [ ('scale',scale),('clf',MLPClassifier(max_iter=350, verbose=False))]
estimator2 = [ ('scale',scale2),('clf',KNeighborsClassifier())] 
estimator3 = [ ('scale',scale),('clf',RandomForestClassifier(random_state=1))]

pipe  = Pipeline(estimator)
pipe2  = Pipeline(estimator2)
pipe3 = Pipeline(estimator3)

param_grid = dict(clf__alpha=np.logspace(-10,-7,4),\
                  clf__hidden_layer_sizes=[ 1500,(500,500),(1000,1000),(1500,1500) ]   ) 
param_grid2 = dict(clf__n_neighbors=[i for i in range(1,30,1)],clf__p= [1,2],clf__weights=['uniform','distance'])
param_grid3 = dict(clf__n_estimators=[100,500,1000],clf__max_depth=[20,50, None],\
                   clf__min_samples_leaf=[1,3,5])


grid = model_selection.GridSearchCV(pipe, param_grid=param_grid,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid2 = model_selection.GridSearchCV(pipe2, param_grid=param_grid2,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid3= model_selection.GridSearchCV(pipe3, param_grid=param_grid3,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')

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

# Starting Cross validation and reporting results
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
   
with open("Log_ip_90Carotapiena"+nome+ ".txt", "w") as text_file:
     print(f"Best results: {best_p}", file=text_file)  
with open("Results_ip_90Carotapiena"+nome+ ".txt", "w") as text_file:
     print(f"Best Scores: {AveRank}", file=text_file)

#BEST ALGORITHMS AND EVALUATION
AveRank.sort(reverse=True)
j=0
for name, pipe in BestEstimator: 
    if AveRank[0][2] is name:
        BestPipe=BestEstimator[j][1]
    j+=1
#Fit the algorithm over all the dataset and save the model
BestPipe.fit(X,y)
dump(BestPipe,'./MLModels/MigliorMod90'+nome )
#


