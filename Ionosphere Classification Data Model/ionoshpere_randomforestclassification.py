'''Compare the training and testing phase performances of three classifiers on Ionosphere dataset. Link to this dataset on UCI ML repository : https://archive.ics.uci.edu/ml/datasets/Ionosphere 

Classifiers to be used: 1. KNN. 2. Decision Tree. 3. Random Forest.

Performance with any five values of K and test ratio 30%, 40% and 50% for KNN.
Performance with any five values of max_depth  and test ratio 30%, 40% and 50% for Decision Trees.
Performance with any five values of n_estimators and test ratio 30%, 40% and 50% for Random Forest.  '''
# Classification template Created on Fri Jun  5 13:34:15 2020
#@author: Yogesh Gopal Palta
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('ionosphere.csv', header=None)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#converting categorical variable to non ategorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

''' Choosing appropriate values of n_estimator'''
error_rate = []
for i in range(1,40):
    
    rfc = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)
    rfc.fit(X_train, y_train)
    pred_i = rfc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. n_estimator Value, Random State-0')
plt.xlabel('n')
plt.ylabel('Error Rate')



''' For given test split and Estimators'''
for i in (30,40,50):
    print("For test Ratio = "+str(i)+"%" )
    for j in (22,24,26,28,32):
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100, random_state = 0)


        # Fitting random forest to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = j, criterion = 'entropy', random_state = 5 )
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        # Confusion Matrix and Scores
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        print("Test Ratio="+str(i)+"% ,n_estimators="+str(j))
        print('Confusion Matrix:')
        print(cm)
        print('F1 Score: '+str(f1))
        print('Accuracy Score: '+str(acc)+'\n')
