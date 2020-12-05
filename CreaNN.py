import os
os.chdir(r'C:\Users\LTM0110User\Desktop\COREMAX')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def MakeNN():
     model=models.Sequential()
     model.add(layers.Dense( 1000, input_dim=177, activation="relu"  ) )
     model.add(layers.Dense( 1000, activation="relu"  ) )
     model.add(layers.Dense( 1, activation="sigmoid"  ) )
     model.compile(loss='binary_cross_entropy', optimizer='adam', metrics='accuracy')
     return(model)