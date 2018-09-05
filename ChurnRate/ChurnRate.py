#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:45:25 2018

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# importing train and test data and storing them in dataframes
train_dataset= pd.read_csv('Train_data.csv')
test_dataset= pd.read_csv('Test_data.csv')
FinalResult_Python = test_dataset[:]


# Removing trailing and leading whitespaces and dots from the data
cat_var= [x for x in train_dataset.dtypes.index if train_dataset.dtypes[x]=='object']
train_dataset[cat_var]= train_dataset[cat_var].apply(lambda x: x.str.strip(' .'))
test_dataset[cat_var]= test_dataset[cat_var].apply(lambda x: x.str.strip(' .'))


# Descriptive statistics of all the numeric features
summary= train_dataset.describe()

# plotting histograms to observe the distributions for all numeric variables
features_numeric = train_dataset.select_dtypes(include = ['float64', 'int64'])
features_numeric.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# plotting bar charts for categorical variables with respect to their 
# distribution in each target class
cat_pred = ['international plan','voice mail plan','area code','state']
for pred in cat_pred:
    crosstab_churn= pd.crosstab(index=train_dataset[pred], columns=train_dataset['Churn'])
    crosstab_churn.reset_index(level=0, inplace= True)
    crosstab_churn[pred]=crosstab_churn[pred].astype('str')

    plot1=plt.bar(x=crosstab_churn[pred], height=(crosstab_churn['False']))
    plot2=plt.bar(x=crosstab_churn[pred], height=(crosstab_churn['True']))
    plt.xlabel(pred)
    plt.xticks(rotation=90)
    plt.ylabel('Churn Count')
    plt.legend((plot1[0],plot2[0]),('False','True'))
    plt.show()
    

# plotting boxplots for all numeric variables for each target class
features_numeric.drop(columns=['area code'], inplace=True)
fig, axes = plt.subplots(4,4, figsize=(15,25)) 
for i,pred in enumerate(features_numeric.columns):
    bxplt = train_dataset.boxplot(column=pred, by="Churn", ax=axes[int(i/4),i%4])
fig.delaxes(axes[3,3]) # remove empty subplot
fig.tight_layout()
plt.show()



# label encoding target class
train_dataset['Churn']= train_dataset['Churn'].map({'False':0, 'True':1})
test_dataset['Churn']= test_dataset['Churn'].map({'False':0, 'True':1})


# plotting correlation matrix heatmap for numeric predictors
corr= train_dataset.corr() 
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, vmax=.5, square=True, annot=True, fmt='0.2f');


# combining dataset to ease preprocessing
train_dataset['source']='train'
test_dataset['source']='test'
dataset= pd.concat([train_dataset,test_dataset],axis=0)


# Finding out the number of missing values in each column
missing_values= dataset.apply(lambda x: sum(x.isnull()))


# label encoding the other binary predictors
map_dict= {'no':0, 'yes':1 ,}
predictors = ['voice mail plan', 'international plan']
for predictor in predictors:
   dataset.loc[:,predictor]=dataset[predictor].map(map_dict)


# creating dummy variabes for nominal variable 'state'
dataset=pd.get_dummies(dataset,columns=['state'],drop_first=True)

# converting and creating dummy variables for nominal variable 'area code'
dataset['area code']=dataset['area code'].astype(str)
dataset=pd.get_dummies(dataset,columns=['area code'],drop_first=True)


# dropping phone number from the dataset
dataset.drop(columns=['phone number'],inplace=True)


# separating the dataset after preprocessing
train_dataset= dataset[dataset['source']=='train']
test_dataset= dataset[dataset['source']=='test']


# separating the predictors and target variable for each train and test dataset
ytrain= train_dataset['Churn']
xtrain= train_dataset.drop(columns=['source','Churn'])
ytest= test_dataset['Churn']
xtest= test_dataset.drop(columns=['source','Churn'])


# storing all predictor names in a variable to be used later
predictors= [x for x in xtrain.columns]


# standardizing both the datasets using the mean and variance of train dataset
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
xtrain= sc.fit_transform(xtrain)
xtest= sc.transform(xtest)
xtrain=pd.DataFrame(data= xtrain, columns = predictors)
xtest=pd.DataFrame(data= xtest, columns = predictors)


'''
Applied PCA to reduce dimensions but the results did not improve, 
so we continued with all the features
from sklearn.decomposition import PCA
pca= PCA(n_components=25, random_state=42)
xtrain_pca = pca.fit_transform(xtrain)
exp_var = pca.explained_variance_ratio_
xtest_pca = pca.transform(xtest)
'''


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# importing warnings to prevent deprecated warnings dialogs from appearing in console
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)



''' Generic function to cross validate and tune hyparameters. It will display the
 train and test scores for multiple evaluation metrics and returns best parameters '''
# algo - classifier object
# x_train - train dataset 
# y_train - target variable corresponding to train dataset
# params - paramgrid containing different combinations to be tested using grid search
# useparams - boolean argument which on False sets param_grid to empty, so that only cross 
# validation is performed 
def model_fit(algo, x_train, y_train, params, useparams):


   
    score_metrics= {'f1score':'f1','precision':'precision','recall':'recall','roc_auc':'roc_auc'}
   

    if(useparams==False):
        params={}
    
    gridsearch= GridSearchCV(algo, params, scoring=score_metrics, cv=5, refit='f1score', verbose=1, return_train_score=True)
    gridsearch= gridsearch.fit(x_train, y_train)
    idx= np.argmax(gridsearch.cv_results_['mean_test_f1score'])

  
    print ("\nModel Report-",str(algo)[0:str(algo).find('(')])
    print("estimator:",gridsearch.best_estimator_)
    print()
    print("Best params:",gridsearch.best_params_)
    print()
    print("Evaluation metrics for best model:")
    print("Train Metrics:")
    print("ROC_AUC :",gridsearch.cv_results_['mean_train_roc_auc'][idx])
    print("PRECISION :",gridsearch.cv_results_['mean_train_precision'][idx])
    print("RECALL :",gridsearch.cv_results_['mean_train_recall'][idx])
    print("F1-SCORE:",gridsearch.cv_results_['mean_train_f1score'][idx])

   
    print("Test Metrics:")
    print("ROC_AUC :",gridsearch.cv_results_['mean_test_roc_auc'][idx])
    print("PRECISION :",gridsearch.cv_results_['mean_test_precision'][idx])
    print("RECALL :",gridsearch.cv_results_['mean_test_recall'][idx])
    print("F1-SCORE :",gridsearch.cv_results_['mean_test_f1score'][idx])

    
   
    return gridsearch.best_estimator_
          


''' Function to plot confusion matrix for test set '''
# algo - classifier object
# ytest - actual values for target variable corresponding to test dataset
# ypred - predicted values for target variable corresponding to test dataset
def plot_confusion_matrix(algo,ytest,ypred):
   
    conf_matrix=confusion_matrix(ytest, ypred)     
    
    sns.heatmap(conf_matrix, fmt='0.2f',annot=True ,xticklabels = ["0", "1"] , yticklabels = ["0", "1"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title(str(algo)[0:str(algo).find('(')])
    
   
    
''' Function to plot ROC_AUC curvealong with threshold for test set  '''  
# ytrue - actual values for target variable corresponding to test dataset
# ypred - predicted probabilities for target variable corresponding to test dataset
def create_roc_curve(ytrue, ypred):
   
   
    # create graph for roc    
    fpr, tpr, thresholds = roc_curve(ytrue, ypred)
    auc_score= auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (auc_score))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
 
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
    plt.show()
   



''' generic function to print feature importance for the best 15 predictors '''
# algo - classifier object
# returns names of top 15 predictors
def print_feature_importance(algo):
    ftimp = pd.Series(algo.feature_importances_, predictors).sort_values(ascending=False)
    ftimp=ftimp.nlargest(n=15)
    plt.figure()
    ftimp.plot(kind='bar', title='_'.join(['feature importance',str(algo)[0:str(algo).find('(')]]))
    return pd.Series(ftimp.index)
    


''' generic function to print Report for the test dataset for different models after the 
 hyperparameters have been tuned '''
# algo - classifier object   
# xtest - test dataset 
# ytest - actual values for target variable corresponding to test dataset 
# returns a Series containing predicted Churn values 
def Test_Set_Report(algo,xtest,ytest):
    
    ypred= algo.predict(xtest)
    ypred_prob= algo.predict_proba(xtest)[:,1]
    
    print ("\nTest Set Report\n")
    print("Algorithm used:",str(algo)[0:str(algo).find('(')])
    print("ROC_AUC:",roc_auc_score(ytest, ypred))
    print("RECALL:",recall_score(ytest,ypred))
    print("PRECISION:",precision_score(ytest,ypred))
    print("F1_SCORE:",f1_score(ytest,ypred))
    print("ACCURACY:",accuracy_score(ytest,ypred))
    
    plot_confusion_matrix(algo, ytest, ypred)
    create_roc_curve(ytest, ypred_prob)
    
    return pd.Series(ypred)



''' Implementing logistic regression with L1 regularization'''

# using class_weight='balanced' to handle imbalanced classes    
from sklearn.linear_model import LogisticRegression

 
logreg= LogisticRegression(class_weight='balanced', penalty='l1', C=0.05, random_state=42)

param_logreg = {'C':[0.01,0.05,0.1,0.2,0.5,0.8,1,2,3]}
logreg=model_fit(logreg, xtrain, ytrain, param_logreg, True)

# train the model with best parameters achieved using gridsearch
logreg.fit(xtrain,ytrain)

# Evaluating the test set with best parameters and printing the final Report
ypred_logreg = Test_Set_Report(logreg,xtest, ytest)




''' Implementing Decision Trees '''

from sklearn.tree import DecisionTreeClassifier

dectree= DecisionTreeClassifier(max_depth=4,min_samples_split=10,max_leaf_nodes=15,random_state=42,class_weight='balanced')

param_tree= { 'max_leaf_nodes':[10,12], 'min_samples_split':[10,15]}
dectree=model_fit(dectree, xtrain, ytrain,param_tree, True)

dectree.fit(xtrain,ytrain)
bestpred_dectree = print_feature_importance(dectree)

ypred_dectree = Test_Set_Report(dectree,xtest, ytest)



''' Random Forest Algorithm '''


from sklearn.ensemble import RandomForestClassifier    

ranfor= RandomForestClassifier(n_estimators=45,max_depth=6,min_samples_split=20,max_leaf_nodes=50,random_state=42, max_features=0.5,class_weight='balanced')

param_ranfor= {'min_samples_split':[20,30,40,50,60],'max_leaf_nodes':[20,30,40,50,60]}
ranfor=model_fit(ranfor, xtrain, ytrain,param_ranfor, False)

ranfor.fit(xtrain,ytrain)
bestpred_rf = print_feature_importance(ranfor)

ypred_ranfor = Test_Set_Report(ranfor,xtest, ytest)


 
''' Gradient Boosting - XGBoost '''
 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import xgboost as xgb
from xgboost.sklearn import XGBClassifier

xgbc= XGBClassifier(n_estimators=300,min_child_weight=1, learning_rate=0.01, scale_pos_weight=9,gamma=3,reg_lambda=0.1,reg_alpha=2, max_depth=10, max_delta_step=2, random_state=42)

param_xgb={'colsample_bytree':[0.7,0.8,0.9,1], 'subsample':[0.9,1]}
xgbc=model_fit(xgbc, xtrain, ytrain, param_xgb, False)

xgbc.fit(xtrain,ytrain)
bestpred_xgb = print_feature_importance(xgbc)

ypred_xgb = Test_Set_Report(xgbc,xtest, ytest)

# xgb.cv is used to get the actual number of n_estimators required based on the learning rate, 
# it uses early_stopping_rounds to get the optimal value
xdtrain = xgb.DMatrix(xtrain,label=ytrain)
cvresult_xgb=xgb.cv(xgbc.get_xgb_params(), xdtrain,nfold=5,num_boost_round=5000,metrics='auc',early_stopping_rounds=50)



''' Storing the predicted results along with the actual prediction for each phone number in a csv file'''

FinalResult_Python = pd.concat([FinalResult_Python,ypred_xgb], axis=1)
FinalResult_Python.rename(columns={0:'Predicted Churn'}, inplace=True)
FinalResult_Python['Predicted Churn']=FinalResult_Python['Predicted Churn'].map({0:'False',1:'True'}) 

FinalResult_Python.to_csv("PredictedChurn_Python.csv", index=False)







