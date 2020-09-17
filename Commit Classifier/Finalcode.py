

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calmap#calender
import squarify#treemap
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold #constantand q constant
from sklearn.preprocessing import StandardScaler #feature scaling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc#accuracy and performance metrics
from scipy import interp #plotting che use hunda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis#?
from sklearn.model_selection import cross_val_score #validation score
import itertools #?

# %matplotlib inline
import warnings #?
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = pd.read_csv('WS_buildaccountdata.csv')

#dataset.head()

print(dataset.shape)

missing_val_count_by_col = dataset.isnull().sum()
#print(type(missing_val_count_by_col))

#bar-plot of the series
plt.figure(figsize=(10,10))
sns.set_style('darkgrid')
plt.title('Missing Value count by column')
missing_val_count_by_col.plot.bar()
plt.show()

#Total count of missing value
total_cells = dataset.size
total_missing = missing_val_count_by_col.sum()
print("Total Missing Value Count: ", total_missing)

# percent of missing data
missing_data = (total_missing/total_cells) * 100
print("Missing data percentage: ", round(missing_data,2),"%")

#filling missig values with median
dataset.fillna(dataset.median(), inplace=True, axis=0)

#count of various categories of tx_state
print(dataset['tx_state'].value_counts())
#encoding categorical data ----> tx_state (using get_dummies)
dataset = pd.get_dummies(dataset, columns=['tx_state', 'label'])
print(dataset.shape)

workFileCol = dict(dataset['tx_working_file_type'].value_counts())
print(workFileCol)
print(len(workFileCol))

#!pip install squarify

#plotting TreeMap for tx_working_file_type
plt.figure(figsize=(12,8), dpi= 80)
labels = list(workFileCol.keys())
sz = list(workFileCol.values())
sz = sz[:51]

squarify.plot(sizes=sz, label=labels, alpha=.8)

plt.title('Treemap of tx_working_file_type')
plt.axis('off')
plt.show()

def fileInclude(f):
  if f not in ['png', 'js', 'nib', 'html', 'java']:
    return 1
  else:
    return 0

#encoding categorical data ----> tx_working_file_type
dataset['WorkingFile_png'] = np.where(dataset['tx_working_file_type'].str.contains('png'), 1, 0)
dataset['WorkingFile_js'] = np.where(dataset['tx_working_file_type'].str.contains('js'), 1, 0)
dataset['WorkingFile_nib'] = np.where(dataset['tx_working_file_type'].str.contains('nib'), 1, 0)
dataset['WorkingFile_html'] = np.where(dataset['tx_working_file_type'].str.contains('html'), 1, 0)
dataset['WorkingFile_java'] = np.where(dataset['tx_working_file_type'].str.contains('java'), 1, 0)
dataset['WorkingFile_others'] = dataset['tx_working_file_type'].apply(lambda file: fileInclude(file))

#drop the original tx_working_file_type column
dataset.drop(labels=['tx_working_file_type'], axis=1, inplace=True)

print(dataset.shape)

#Parsing dates
print(dataset['dt_submit_date'].head())

#method --> to_datetime
dataset['dt_submit_date'] = pd.to_datetime(dataset['dt_submit_date'], format = "%Y-%m-%d")

#checking new data type
print(dataset['dt_submit_date'].head())

dataset['submit_year'] = dataset['dt_submit_date'].dt.year
dataset['submit_month'] = dataset['dt_submit_date'].dt.month
dataset['submit_weekday'] = dataset['dt_submit_date'].dt.weekday

#!pip install calmap

dataset2 = dataset.set_index('dt_submit_date')
dataset2.head()

#Calendar heatmap
plt.figure(figsize=(16,10), dpi= 80)
calmap.calendarplot(dataset2['submit_year'], yearascending=True, fig_kws={'figsize': (16,10)}, yearlabel_kws={'color':'black', 'fontsize':14}, subplot_kws={'title':'Commits'})
plt.show()

#WEEKDAYS AND MONTH
fig, ax = plt.subplots(1, 2, figsize=(18,4))

wkday = dataset['submit_weekday'].values
mnth = dataset['submit_month'].values
sns.set_style('darkgrid')

sns.distplot(wkday,ax=ax[0], color='g')
ax[0].set_title('Distribution over the weekdays', fontsize=14)
ax[0].set_xlim([min(wkday), max(wkday)])
ax[0].set_xlabel('weekdays')
ax[0].set_ylabel('Number of commits')

sns.distplot(mnth, ax=ax[1], color='brown')
ax[1].set_title('Distribution over the month', fontsize=14)
ax[1].set_xlim([min(mnth), max(mnth)])
ax[1].set_xlabel('Months')
ax[1].set_ylabel('Number of commits')

plt.show()

#YEAR AND MONTH
fig, ax = plt.subplots(1, 2, figsize=(18,4))

yr = dataset['submit_year'].values
mnth = dataset['submit_month'].values
sns.set_style('darkgrid')

sns.distplot(yr,ax=ax[0], color='r')
ax[0].set_title('Distribution over the years', fontsize=14)
ax[0].set_xlim([min(yr), max(yr)])
ax[0].set_xlabel('Years')
ax[0].set_ylabel('Number of commits')


sns.distplot(mnth, ax=ax[1], color='b')
ax[1].set_title('Distribution of commit time', fontsize=14)
ax[1].set_xlim([min(mnth), max(mnth)])
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Number of commits')

plt.show()

print(dataset['tm_submit_time'].head())
dataset['tm_submit_time'] = pd.to_datetime(dataset['tm_submit_time'], format = "%H:%M:%S")
print(dataset['tm_submit_time'].head())

dataset['submit_hour'] = dataset['tm_submit_time'].dt.hour
dataset['submit_minute'] = dataset['tm_submit_time'].dt.minute

#HOURS AND MINUTES
fig, ax = plt.subplots(1, 2, figsize=(18,4))

hr = dataset['submit_hour'].values
mint = dataset['submit_minute'].values
sns.set_style('darkgrid')

sns.distplot(hr,ax=ax[0], color='r')
ax[0].set_title('Distribution over the hours', fontsize=14)
ax[0].set_xlim([min(hr), max(hr)])
ax[0].set_xlabel('Hours')
ax[0].set_ylabel('Number of commits')


sns.distplot(mint, ax=ax[1], color='b')
ax[1].set_title('Distribution over minutes', fontsize=14)
ax[1].set_xlim([min(mint), max(mint)])
ax[1].set_xlabel('Minutes')
ax[1].set_ylabel('Number of commits')


plt.show()

#droping columns
dataset.drop(labels=['dt_submit_date', 'tm_submit_time', 'submit_year'], axis=1, inplace=True)

print(dataset.shape)

#checking skew of every column
dataset_reduced = dataset.iloc[:, 3:]
for j in dataset_reduced.columns:

    dataset_reduced[j] = dataset_reduced[j].map(lambda i: np.log(i) if i > 0 else 0) 
    print(j,": ",dataset_reduced[j].skew())

#dividing the dataset into dependent and independent variable
x = dataset_reduced
x.drop(labels=['label_False', 'label_True'],axis=1, inplace=True)
y = dataset['label_True']

x.shape

#Visualising outliers
f,ax1 = plt.subplots(1, 1, figsize=(20,6))
sns.set_style('darkgrid')
colors = ['#B3F9C5', '#f9c5b3']
sns.boxplot(x='label_True', y= 'nu_halstead_total', data = dataset, orient='v', palette=colors, ax=ax1)
plt.title('Outlier visualisation for nu_halstead_total')
plt.show()

#Visualising outliers
f,ax1 = plt.subplots(1, 1, figsize=(20,6))
#sns.set_style('darkgrid')
colors = ['#B3F9C5', '#f9c5b3']
sns.boxplot(x='label_True', y= 'nu_filesize_total', data = dataset, orient='v', palette=colors, ax=ax1)
plt.title('Outlier visualisation for nu_filesize_total')
plt.show()

#Visualising outliers
f,ax1 = plt.subplots(1, 1, figsize=(20,6))
#sns.set_style('darkgrid')
colors = ['#B3F9C5', '#f9c5b3']
sns.boxplot(x='label_True', y= 'nu_cyclo_total', data = dataset, orient='v', palette=colors, ax=ax1)
plt.title('Outlier visualisation for nu_cyclo_total')
plt.show()

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify=y, random_state=42)

print(x_train.shape, y_train.shape)

#Removing constant features
constant_filter = VarianceThreshold(threshold=0)

#apply this filter to x_train
constant_filter.fit(x_train)

#get all ******NON CONSTANT**** features
print(len(x_train.columns[constant_filter.get_support()]))

# list of all constant columns
constant_columns = [column for column in x_train.columns if column not in x_train.columns[constant_filter.get_support()]]
print(len(constant_columns))
print(constant_columns)

#removing constant features
x_train.drop(labels=constant_columns, axis=1, inplace=True)
x_test.drop(labels=constant_columns, axis=1, inplace=True)

print(x_train.shape, x_test.shape)

#Removing quasi-constant features
qconstant_filter = VarianceThreshold(threshold=0.01)

#apply this filter to x_train
qconstant_filter.fit(x_train)

#get all ******NON CONSTANT**** features
print(len(x_train.columns[qconstant_filter.get_support()]))

# list of all constant columns
qconstant_columns = [column for column in x_train.columns if column not in x_train.columns[qconstant_filter.get_support()]]
print(len(qconstant_columns))
print(qconstant_columns)

#removing constant features
x_train.drop(labels=qconstant_columns, axis=1, inplace=True)
x_test.drop(labels=qconstant_columns, axis=1, inplace=True)

print(x_train.shape, x_test.shape)

#Checking correlation visualization -method2
def correlation_heatmap(dt):
    correlations = dt.corr()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .70}, cmap="YlGnBu")
    plt.show()
# vmaxto anchor the colormap

correlation_heatmap(x_train)

#Removing correlated features in x_train
correlated_features_train = set()
correlation_matrix_train = x_train.corr()

for i in range(len(correlation_matrix_train .columns)):
    for j in range(i):
        if abs(correlation_matrix_train.iloc[i, j]) > 0.8:
            colname = correlation_matrix_train.columns[i]
            correlated_features_train.add(colname)

print(len(correlated_features_train))
print(correlated_features_train)

#Removing correlated features in x_test
correlated_features_test = set()
correlation_matrix_test = x_train.corr()

for i in range(len(correlation_matrix_test .columns)):
    for j in range(i):
        if abs(correlation_matrix_test.iloc[i, j]) > 0.8:
            colname = correlation_matrix_test.columns[i]
            correlated_features_test.add(colname)

print(len(correlated_features_test))
print(correlated_features_test)

#Dropping the related columns
x_train.drop(labels=correlated_features_train, axis=1, inplace=True)
x_test.drop(labels=correlated_features_test, axis=1, inplace=True)

print(x_train.shape, x_test.shape)

#Check the label count
dataset['label_True'].value_counts()

#See the label percentage of the dataset
print('False', round(dataset['label_True'].value_counts()[0]/len(dataset) * 100,2), '% of the dataset')
print('True', round(dataset['label_True'].value_counts()[1]/len(dataset) * 100,2), '% of the dataset')

#Label distribution plot
plt.bar(['False','True'], dataset['label_True'].value_counts(), color=['teal','grey'])
plt.xlabel('label')
plt.ylabel('Number of commits')
plt.title('Class Distributions \n False: Developer Activity \n True: Build Account Activity', fontsize=14)
plt.annotate('{}\n({:.4}%)'.format(dataset['label_True'].value_counts()[0], dataset['label_True'].value_counts()[0]/dataset['label_True'].count()*100),
             (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}\n({:.4}%)'.format(dataset['label_True'].value_counts()[1], dataset['label_True'].value_counts()[1]/dataset['label_True'].count()*100),
             (0.70, 0.45), xycoords='axes fraction')
plt.tight_layout()
plt.show()

#A function to feature scale the variables 
def feature_scaling(X, X_test=x_test):
    std_scale = StandardScaler().fit(X)
    X_std = std_scale.transform(X)
    X_test_std = std_scale.transform(X_test)
    return X_std, X_test_std

#Applying feature scaling
x_std, x_test_std = feature_scaling(x_train)

#Define a list of classifier models
classifiers = []

#classifiers.append(('Logistic Regression', LogisticRegression(random_state=42)))
#classifiers.append(('Naive Bayes', GaussianNB()))
##classifiers.append(('KNN', KNeighborsClassifier()))#This one takes a very long time to run!
##classifiers.append(('SVM', SVC(random_state=42, probability=True))) #This one takes a very long time to run!
#classifiers.append(('Decision Tree', DecisionTreeClassifier(random_state=42)))
#classifiers.append(('Random Forest', RandomForestClassifier(random_state=42)))
classifiers.append(('LDA', LinearDiscriminantAnalysis()))
#classifiers.append(('QDA', QuadraticDiscriminantAnalysis()))

#Ensemble classifier - All classifiers have the same weight
eclf = VotingClassifier(estimators=classifiers, voting='soft', weights=np.ones(len(classifiers)))

#A helper function to plot confusion matrix
sns.set_style('dark')
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#A function to plot Receiver Operating Characteristic Curve and Confusion Matrix
def plot_CM_and_ROC_curve(classifier, X_train, y_train, X_test, y_test):
    name = classifier[0]
    classifier = classifier[1]

    mean_fpr = np.linspace(0, 1, 100)
    class_names = ['False', 'True']
    confusion_matrix_total = [[0, 0], [0, 0]]
    
    #Obtain probabilities for each class
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    print (probas_)
    print (len(probas_))
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=1, color='b', label='ROC (AUC = %0.7f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve - model: ' + name)
    plt.legend(loc="lower right")
    plt.show()
    
    #Store the confusion matrix result to plot a table later
    y_pred=classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    confusion_matrix_total += cnf_matrix
    print(confusion_matrix_total)    
    #Print precision and recall
    tn, fp = confusion_matrix_total.tolist()[0]
    fn, tp = confusion_matrix_total.tolist()[1]

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = (2 * precision * recall)/(precision + recall)

    print('Accuracy = {:2.2f}%'.format(accuracy*100))
    print('Precision = {:2.2f}%'.format(precision*100))
    print('Recall = {:2.2f}%'.format(recall*100))
    print('F1 Score = {:2.2f}'.format(fscore))
    
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix_total, classes=class_names, title='Confusion matrix - model: ' + name)
    plt.show()

#Iterate over the list of classifiers and train different models
for clf in classifiers:
  plot_CM_and_ROC_curve(clf, x_std, y_train, x_test_std, y_test)

#Ensemble Model
plot_CM_and_ROC_curve(('Ensemble model', eclf), x_std, y_train, x_test_std, y_test)

#A list of classifiers to run K-Fold Cross Validation on
clfrs = []

clfrs.append(('Logistic Regression', LogisticRegression(random_state=42)))
clfrs.append(('Naive Bayes', GaussianNB()))
#classifiers.append(('KNN', KNeighborsClassifier()))#This one takes a very long time to run!
#classifiers.append(('SVM', SVC(random_state=42, probability=True))) #This one takes a very long time to run!
clfrs.append(('Decision Tree', DecisionTreeClassifier(random_state=42)))
clfrs.append(('Random Forest', RandomForestClassifier(random_state=42)))
clfrs.append(('LDA', LinearDiscriminantAnalysis()))
clfrs.append(('QDA', QuadraticDiscriminantAnalysis()))
clfrs.append(('Ensemble Model', eclf))

#Iterate over the list to validate every model
#This step of validating every trained model takes a lot of time to execute. As the dataset it has to validate over is very large
#The runtime of the code is subject to a good GPU unit, which in general laptops is a constraint
#The value of k is set 20
for classifier in clfrs:
    clf = classifier[1]
    clf.fit(x_train, y_train)
    training_score = cross_val_score(clf, x_train, y_train, cv=20)
    print("Classifiers: ", classifier[0], "has a cross validation score of", round(training_score.mean(), 2) * 100, "% accuracy score")

#A bar plot for all the trained models and their F1 score
train_accuracies = [0.71, 0.61, 1.00, 1.00, .72, 0.68, 0.86]
models = ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'LDA', 'QDA', 'Ensemble']
fig = plt.figure(figsize = (10, 5))
plt.bar(models, train_accuracies, color ='green', width = 0.4) 
  
plt.xlabel("Predictive Models") 
plt.ylabel("F1 Score") 
plt.title("Performance of various models") 
plt.show()