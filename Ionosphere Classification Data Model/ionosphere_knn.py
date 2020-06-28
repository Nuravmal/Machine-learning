#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score





#importing dataset
#use header=None as dataset have no headings
dataset=pd.read_csv('ionosphere.data',header=None)
#to rename column name of dependent variable
dataset.rename(columns={34:'target'})


#plotting whole dataset isnt possible so we will pick some random column with target value
grr=pd.plotting.scatter_matrix(dataset.iloc[:,[0,1,2,32,33,34]],marker='o',figsize=(25,10))


#checking for missing value
print(dataset.isnull())


#converting dataframe into arrays
#all predictors or independent variables are in X whereas y is dependent variable or target value
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]



#converting categorical variable to non ategorical variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



#defining function to perform elbow method to calculate correct value of k
def elbow_method(X_train,y_train):
        
    error_rate = []
    for i in range(1,40):
        
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')



f1score=[]
accuracyscore=[]

"""test ratio split :30%"""

#splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#using elbow method to choose correct value of k
elbow_method(X_train,y_train)

#from above plot it is clear that error rate is least for k=2 and then k=4 then k=6
#we have to take 5 values of k
k_30=[2,3,4,5,7]
k=k_30
for i in k:
     #fitting model with training set
     classifier=KNeighborsClassifier(n_neighbors=i)
     classifier.fit(X_train,y_train)
     print("*********Prediction for K= "+str(i)+"*********")
     #predicting with model
     y_pred = classifier.predict(X_test)
     #computing confusion matrix
     cm = confusion_matrix(y_test, y_pred)
     print("Confusion Matrix(Test Ratio=30%) for K= "+str(i)+ " is: ",cm)
     #computing f1_score
     f1=f1_score(y_test,y_pred)
     print("f1_score(Test Ratio=30%) for K= "+str(i)+" is: ",f1)
     f1score.append(f1);
     #computing accuracy score
     accuracy=accuracy_score(y_test,y_pred)
     print("accuracy_score(Test Ratio=30%) for K="+str(i)+" is: ",accuracy)
     accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(k,f1score,color='blue', label="Test Ratio 30%",marker='o')
plt.title('K Value Vs f1 Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(k,accuracyscore,color='blue', label="Test Ratio 30%",marker='o')
plt.title('K Value Vs Accuracy Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy score')
   

"""test ratio split :40%"""

#splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


#using elbow method to choose correct value of k
elbow_method(X_train,y_train)

#from above plot it is clear that error rate is least for k=2 and then k=4 then k=6 then 8

#we have to take 5 values of k
k_40=[2,3,4,8,11]
k=k_40
for i in k:
     #fitting model with training set
     classifier=KNeighborsClassifier(n_neighbors=i)
     classifier.fit(X_train,y_train)
     print("*********Prediction for K= "+str(i)+"*********")
     #predicting with model
     y_pred = classifier.predict(X_test)
     #computing confusion matrix
     cm = confusion_matrix(y_test, y_pred)
     print("Confusion Matrix(Test Ratio=40%) for K= "+str(i)+ " is: ",cm)
     #computing f1_score
     f1=f1_score(y_test,y_pred)
     print("f1_score(Test Ratio=40%) for K= "+str(i)+" is: ",f1)
     f1score.append(f1);
     #computing accuracy score
     accuracy=accuracy_score(y_test,y_pred)
     print("accuracy_score(Test Ratio=40%) for K="+str(i)+" is: ",accuracy)
     accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(k,f1score[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.title('K Value Vs f1 Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(k,accuracyscore[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.title('K Value Vs Accuracy Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy score')
   


"""test ratio split :50%"""

#splitting dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


#using elbow method to choose correct value of k
elbow_method(X_train,y_train)

#from above plot it is clear that error rate is least for k=2 and then k=4 then k=6 then 8

#we have to take 5 values of k
k_50=[2,4,6,8,10]
k=k_50
for i in k:
     #fitting model with training set
     classifier=KNeighborsClassifier(n_neighbors=i)
     classifier.fit(X_train,y_train)
     print("*********Prediction for K= "+str(i)+"*********")
     #predicting with model
     y_pred = classifier.predict(X_test)
     #computing confusion matrix
     cm = confusion_matrix(y_test, y_pred)
     print("Confusion Matrix(Test Ratio=50%) for K= "+str(i)+ " is: ",cm)
     #computing f1_score
     f1=f1_score(y_test,y_pred)
     print("f1_score(Test Ratio=50%) for K= "+str(i)+" is: ",f1)
     f1score.append(f1);
     #computing accuracy score
     accuracy=accuracy_score(y_test,y_pred)
     print("accuracy_score(Test Ratio=50%) for K="+str(i)+" is: ",accuracy)
     accuracyscore.append(accuracy)

#plotting f1_score
     
plt.figure(figsize=(15,15))
plt.plot(k,f1score[10:15],color='blue', label="Test Ratio 50%",marker='o')
plt.title('K Value Vs f1 Score')
plt.legend()
plt.xlabel('K')
plt.ylabel('f1 score')

#plotting acuracy score

plt.figure(figsize=(15,15))
plt.plot(k,accuracyscore[10:15],color='blue', label="Test Ratio 50%",marker='o')
plt.title('K Value Vs Accuracy Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy score')
   

#final plot
#f1 plot
plt.figure(figsize=(15,15))
plt.plot(k_30,f1score[0:5],color='red', label="Test Ratio 30%",marker='o')
plt.plot(k_40,f1score[5:10],color='blue', label="Test Ratio 40%",marker='o')
plt.plot(k_50,f1score[10:15],color='green', label="Test Ratio 50%",marker='o')
plt.title('K Vs f1 Score with test ratios')
plt.legend()
plt.xlabel('K')
plt.ylabel('f1 score')
