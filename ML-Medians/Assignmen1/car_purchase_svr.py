import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("Car_Purchasing_Data.csv",encoding='ISO-8859-1')

X=df.iloc[:,3:8].values
y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y= sc_y.fit_transform(y.reshape(-1,1))
"""
# Fitting the SVR Regression Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred=regressor.predict(X_test)

from sklearn.metrics import r2_score 
print(r2_score(y_pred,y_test))
