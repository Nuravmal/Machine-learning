# -*- coding: utf-8 
# Ionosphere_ Decision Tree 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



#Importing the dataset
dataset=pd.read_csv('ionosphere.data',header=None)
#checking for missing value
print(dataset.isnull())

#converting dataframe into arrays
#all predictors or independent variables are in X whereas y is dependent variable or target value
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#converting categorical variable to non ategorical variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#defining function to perform elbow method to calculate correct value of k
def elbow_method(X_train,y_train):
        
    error_rate = []
    for i in range(1,50):
        dec_tree= DecisionTreeClassifier(max_depth=i)
        dec_tree.fit(X_train, y_train)
        pred_i= dec_tree.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. Max_depth Value  ')
    plt.xlabel('n')
    plt.ylabel('Error Rate')

f1score=[]
accuracyscore=[]

"""test ratio :30%"""

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state = 0)

#using elbow method to choose correct value of k
elbow_method(X_train,y_train)
#from above plot it is clear that error rate is least for max_depth=1,5,17 etc
#we have to take 5 values of max_Depth
MAX_DEPTH_30=[2,6,18,20,48]
MAX_DEPTH=MAX_DEPTH_30
for i in range(len(MAX_DEPTH)):
    #fitting model with training set
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i])
    dec_tree.fit(X_train,y_train)
    print("*********Prediction for Max_depth= "+str(MAX_DEPTH[i])+"*********")
    #predicting with model
    y_pred = dec_tree.predict(X_test)
    #computing confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix(Test Ratio=30%) for max_depth= "+str(MAX_DEPTH[i])+ " is: ",cm)
    #computing f1_score
    f1=f1_score(y_test,y_pred)
    print("f1_score(Test Ratio=30%) for K= "+str(MAX_DEPTH[i])+" is: ",f1)
    f1score.append(f1);
    #computing accuracy score
    accuracy=accuracy_score(y_test,y_pred)
    print("accuracy_score(Test Ratio=30%) for K="+str(MAX_DEPTH[i])+" is: ",accuracy)
    accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,f1score,color='blue', label="Test Ratio 30%",marker='o')
plt.title('Max_depth Value Vs f1 Score for Test Ratio=30%')
plt.legend()
plt.xlabel('Max_Depth')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,accuracyscore,color='blue', label="Test Ratio 30%",marker='o')
plt.title('Max_depth Vs Accuracy Score for Test Ratio=30%')
plt.legend()
plt.xlabel('Max_depth')
plt.ylabel('Accuracy score')
    

"""test ratio :40%"""

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state = 0)

MAX_DEPTH_40=[2,6,18,20,48]
MAX_DEPTH=MAX_DEPTH_40
for i in range(len(MAX_DEPTH)):
    #fitting model with training set
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i])
    dec_tree.fit(X_train,y_train)
    print("*********Prediction for Max_depth= "+str(MAX_DEPTH[i])+"*********")
    #predicting with model
    y_pred = dec_tree.predict(X_test)
    #computing confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix(Test Ratio=40%) for max_depth= "+str(MAX_DEPTH[i])+ " is: ",cm)
    #computing f1_score
    f1=f1_score(y_test,y_pred)
    print("f1_score(Test Ratio=40%) for K= "+str(MAX_DEPTH[i])+" is: ",f1)
    f1score.append(f1);
    #computing accuracy score
    accuracy=accuracy_score(y_test,y_pred)
    print("accuracy_score(Test Ratio=40%) for K="+str(MAX_DEPTH[i])+" is: ",accuracy)
    accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,f1score[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.title('Max_depth Value Vs f1 Score for Test Ratio=40%')
plt.legend()
plt.xlabel('Max_Depth')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,accuracyscore[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.title('Max_depth Vs Accuracy Score for Test Ratio=40%')
plt.legend()
plt.xlabel('Max_depth')
plt.ylabel('Accuracy score')

"""test ratio :50%"""

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50,random_state = 0)
MAX_DEPTH_50=[2,6,18,20,48]
MAX_DEPTH=MAX_DEPTH_50

for i in range(len(MAX_DEPTH)):
    #fitting model with training set
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i])
    dec_tree.fit(X_train,y_train)
    print("*********Prediction for Max_depth= "+str(MAX_DEPTH[i])+"*********")
    #predicting with model
    y_pred = dec_tree.predict(X_test)
    #computing confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix(Test Ratio=50%) for max_depth= "+str(MAX_DEPTH[i])+ " is: ",cm)
    #computing f1_score
    f1=f1_score(y_test,y_pred)
    print("f1_score(Test Ratio=50%) for K= "+str(MAX_DEPTH[i])+" is: ",f1)
    f1score.append(f1);
    #computing accuracy score
    accuracy=accuracy_score(y_test,y_pred)
    print("accuracy_score(Test Ratio=50%) for K="+str(MAX_DEPTH[i])+" is: ",accuracy)
    accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,f1score[10:15],color='blue', label="Test Ratio 50%",marker='o')
plt.title('Max_depth Value Vs f1 Score for Test Ratio=50%')
plt.legend()
plt.xlabel('Max_Depth')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH,accuracyscore[10:15],color='blue', label="Test Ratio 50%",marker='o')
plt.title('Max_depth Vs Accuracy Score for Test Ratio=50%')
plt.legend()
plt.xlabel('Max_depth')
plt.ylabel('Accuracy score')


#final plot
#f1 plot
plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH_30,f1score[0:5],color='red', label="Test Ratio 30%",marker='o')
plt.plot(MAX_DEPTH_40,f1score[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.plot(MAX_DEPTH_50,f1score[10:15],color='green', label="Test Ratio 50%",marker='o')
plt.title('MAX_DEPTH Vs f1 Score with test ratios')
plt.legend()
plt.xlabel('MAX_DEPTH')
plt.ylabel('f1 score')

#Accuracy plot
plt.figure(figsize=(15,15))
plt.plot(MAX_DEPTH_30,accuracyscore[0:5],color='red', label="Test Ratio 30%",marker='o')
plt.plot(MAX_DEPTH_40,accuracyscore[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.plot(MAX_DEPTH_50,accuracyscore[10:15],color='green', label="Test Ratio 50%",marker='o')
plt.title('MAX_DEPTH Vs Accuracy Score with test ratios')
plt.legend()
plt.xlabel('MAX_DEPTH')
plt.ylabel('Accuracy score')
