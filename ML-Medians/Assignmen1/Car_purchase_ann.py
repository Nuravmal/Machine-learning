# Regression 
import pandas as pd
import numpy as np
import tensorflow as tf


df=pd.read_csv("Car_Purchasing_Data.csv",encoding='ISO-8859-1')

X=df.iloc[:,3:8].values
y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer="adam",loss="mean_squared_error")

ann.fit(X_train,y_train,batch_size=10,epochs=150)
y_pred=ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate(y_pred.reshape(len(y_pred),1)),y_test.reshape(len(y_pred),1))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))