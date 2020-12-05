 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

X=np.expand_dims(X,axis=2)
Xt=np.expand_dims(Xt,axis=2)

model = models.Sequential()
model.add(layers.Conv1D(32, 3 , activation='relu', input_shape=(396,1 )))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, (3), activation='relu'))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, (3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10, 
                    validation_data=(Xt, yt))
yp=model.predict(Xt)
ypred=[]
for i in yp:
    ypred.append(np.argmax(i))
ypred=np.array(ypred, dtype='float64')

Accuracy=metrics.accuracy_score(yt, ypred)



Accuracy_Bal=metrics.balanced_accuracy_score(yt, fisso) 
yp2=pipe.predict(DBX)
fisso2=yp2.astype(int)
Accuracy_Blind=metrics.accuracy_score(DBY,yp2)

