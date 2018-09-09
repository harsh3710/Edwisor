#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:32:49 2018

"""


# importing required libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# importing dataset and storing as dataframe
dataset = pd.read_csv("day.csv")
dataset_copy = dataset[:]

# dropping dteday and instant variable
dataset.drop(columns=['instant','dteday'],axis=1, inplace=True)


#convert discreet variables  to categorical:
cat_features = ['season','yr','mnth','holiday','workingday','weekday','weathersit']
for ftr in cat_features:
    dataset[ftr] = dataset[ftr].astype(str)
    print(dataset[ftr].value_counts())
    
    
# mapping numerical categories to labels for better representation
dataset['season'] = dataset['season'].map({'1':'spring','2':'summer','3':'fall','4':'winter'})
dataset['weathersit'] = dataset['weathersit'].map({'1':'clear/partly_cloudy','2':'mist/cloudy','3':'light_snow/rain','4':'heavy_rain/snow'})
    

# Missing value analysis
missing_values = dataset.apply(lambda x: sum(x.isnull()))


# descriptive summary of the numerical variables
data_summary = dataset.describe()


# observing the distribution of target variable using boxplots and histograms
target_var = dataset.select_dtypes(include = ['int64'])


fig, axes = plt.subplots(2,3, figsize=(15,10)) 
for i,pred in enumerate(target_var.columns):
    bxplt = sns.boxplot(orient='v',data= target_var, y=pred, ax=axes[0,i])
    plt.sca(axes[1,i])
    plt.hist(dataset[pred],bins=30)
    plt.xlabel(pred)
 
    

# plotting histograms to observe the distributions for all numeric variables
features_numeric = dataset.select_dtypes(include = ['float64'])
features_numeric.hist(figsize=(10, 10), bins=40, xlabelsize=8, ylabelsize=8);



# plotting scatterplos to observe relation between continous variables
fig, axes = plt.subplots(4,3, figsize=(20,20)) 
for i,pred in enumerate(features_numeric.columns):
    plt.sca(axes[int(i%4),0])
    plt.scatter(x=pred,y='casual',data=dataset)
    plt.xlabel(pred)
    plt.ylabel('casual')
    plt.sca(axes[int(i%4),1])
    plt.scatter(x=pred,y='registered',data=dataset)
    plt.xlabel(pred)
    plt.ylabel('registered')
    plt.sca(axes[int(i%4),2])
    plt.scatter(x=pred,y='cnt',data=dataset)
    plt.xlabel(pred)
    plt.ylabel('cnt')
    



# plotting boxplots for casual and registerd users for categorical variables
fig, axes = plt.subplots(7,3, figsize=(15,25)) 
for i,pred in enumerate(cat_features):
    bxplt1 = sns.boxplot(data= dataset, x=pred,y=dataset['casual'], ax=axes[int(i%7),0])
    bxplt1.set(xlabel=pred)
    bxplt2 = sns.boxplot(data= dataset, x=pred,y=dataset['registered'], ax=axes[int(i%7),1])
    bxplt2.set(xlabel=pred)
    bxplt3 = sns.boxplot(data= dataset, x=pred,y=dataset['cnt'], ax=axes[int(i%7),2])
    bxplt3.set(xlabel=pred)
    
    
    
# plotting correlation matrix heatmap for numeric predictors
corr= dataset.corr() 
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, center=1, square=True, annot=True, fmt='0.3f');


# one hot encoding the below features
dataset = pd.get_dummies(dataset,dtype=int, drop_first=True, columns=['season','weathersit','mnth','weekday'])
dataset[['yr','holiday','workingday']] = dataset[['yr','holiday','workingday']].astype(int)

# dropping temp feature and keeping atemp since both are highly correlated
dataset.drop(columns=['temp'],axis=1, inplace=True)

# dropping observation containing humidity=0 which is next to impossible
temp_data = dataset[dataset['hum']!=0]

# dropping observations containing outliers for casual variable for better accuracy
temp_data = temp_data.loc[(temp_data['casual']-temp_data['casual'].mean())<=(3*temp_data['casual'].std())]


# splitting dataset into train and test  
train ,test = train_test_split(temp_data, test_size= 0.1, random_state=42)
 


# further splitting the dataset to separate the target variable from the predictors
FinalResult_Python= dataset_copy.loc[pd.Series(test.index),]
ytrain_casual = train['casual']
ytrain_registered = train['registered']
ytrain_count = train['cnt']
xtrain = train.drop(columns=['casual','registered','cnt'], axis=1)	


ytest_casual = test['casual']
ytest_registered = test['registered']
ytest_count = test['cnt']
ytest_count=ytest_count.reset_index()
ytest_count.drop(columns='index',axis=1, inplace=True)
xtest = test.drop(columns=['casual','registered','cnt'], axis=1)	
xtest=xtest.reset_index()
xtest.drop(columns='index',axis=1, inplace=True)


# storing all predictor names in a variable to be used later
predictors = [x for x in xtrain.columns]



''' Generic function to cross validate and tune hyparameters. It will display the
 train and test scores for evaluation metric RMSE and returns best parameters '''
# algo - regression object
# x_train - train dataset 
# y_train - target variable corresponding to train dataset
# params - paramgrid containing different combinations to be tested using grid search
# useparams - boolean argument which on False sets param_grid to empty, so that only cross 
# validation is performed 
def model_fit(algo, x_train, y_train, params, useparams):

    if(useparams==False):
        params={}
    
    gridsearch= GridSearchCV(algo, params, scoring='neg_mean_squared_error', cv=5, refit='neg_mean_squared_error', verbose=1, return_train_score=True)
    gridsearch= gridsearch.fit(x_train, y_train)
    
  
    print ("\nModel Report-",str(algo)[0:str(algo).find('(')])
    print("estimator:",gridsearch.best_estimator_)
    print()
    print("Best params:",gridsearch.best_params_)
    print()
    print("Evaluation metrics for best model:")
    print("Train Metrics:")
    print("Root Mean Squared Error :",np.sqrt(np.abs(gridsearch.cv_results_['mean_train_score'].max())),"\n")
    print("Test Metrics:")
    print("Root Mean Squared Error :",np.sqrt(np.abs(gridsearch.best_score_)),"\n")
   
    return gridsearch.best_estimator_

    

''' generic function to print Report for the test dataset for different models after the 
 hyperparameters have been tuned '''
# algo - regression object   
# xtest - test dataset 
# ytest - actual values for target variable corresponding to test dataset 
# returns a Series containing predicted count values 
def Test_Set_Report(algoname,algo):
    
    algo.fit(xtrain,ytrain_casual)
    ypred_casual= algo.predict(xtest)
    
    algo.fit(xtrain,ytrain_registered)
    ypred_registered= algo.predict(xtest)
    
    ypred_count = ypred_casual + ypred_registered
    
    print ("\nTest Set Report\n")
    print("Algorithm used:",algoname)
    print("RMSE:",np.sqrt(mean_squared_error(ytest_count, ypred_count)))
    print("RMSLE",np.sqrt(mean_squared_log_error(ytest_count,(ypred_count))))
    
    plt.scatter(ytest_count,ypred_count)
    
    return pd.Series(ypred_count)


''' generic function to print feature importance for the best 15 predictors '''
# algo - regression object
# returns names of top 15 predictors
def print_feature_importance(algo):
    
    algo.fit(xtrain,ytrain_casual)
    ftimp1 = pd.Series(algo.feature_importances_, predictors).sort_values(ascending=False)
    ftimp1=ftimp1.nlargest(n=15)
    plt.figure()
    ftimp1.plot(kind='bar', title='_'.join(['feature importance- Casual users',str(algo)[0:str(algo).find('(')]]))

    algo.fit(xtrain,ytrain_registered)
    ftimp2 = pd.Series(algo.feature_importances_, predictors).sort_values(ascending=False)
    ftimp2=ftimp2.nlargest(n=15)
    plt.figure()
    ftimp2.plot(kind='bar', title='_'.join(['feature importance- Registered users',str(algo)[0:str(algo).find('(')]]))
    
    return pd.concat([pd.Series(ftimp1.index),pd.Series(ftimp2.index)], axis=1)




''' LINEAR REGRESSION '''

linreg = LinearRegression()

linreg = model_fit(linreg, xtrain, (ytrain_casual),{},False)
linreg = model_fit(linreg, xtrain, (ytrain_registered),{},False)

ypred_lr_count = Test_Set_Report("Linear Regression",linreg)

result_linear = pd.concat([ytest_count,pd.Series(ypred_lr_count)], axis=1)



''' REGULARIZED LINEAR REGRESSION - RIDGE'''
ridge = Ridge(random_state=42, alpha=0.5)
param_ridge = {'alpha':[0.1,0.2,0.3,0.4,0.5,0.7,1,1.2,1.5,1.8,2]}

ridge = model_fit(ridge,xtrain,ytrain_registered,param_ridge,False)
ridge = model_fit(ridge,xtrain,ytrain_casual,param_ridge,False)

ypred_ridge_count = Test_Set_Report("Regularized Linear Regression - Ridge",ridge)

result_ridge = pd.concat([ytest_count,pd.Series(ypred_ridge_count)], axis=1)



''' REGULARIZED LINEAR REGRESSION - LASSO'''
lasso = Lasso(random_state=42, alpha=0.4)
param_lasso = {'alpha':[0.1,0.2,0.3,0.4,0.5,0.7,1,1.2,1.5,1.8,2]}

lasso = model_fit(lasso,xtrain,(ytrain_casual),param_lasso,False)
lasso = model_fit(lasso,xtrain,(ytrain_registered),param_lasso,False)

ypred_lasso_count = Test_Set_Report("Regularized Linear Regression - Lasso",lasso)

result_lasso = pd.concat([ytest_count,pd.Series(ypred_lasso_count)], axis=1)



''' DECISION TREE REGRESSOR '''

dectree = DecisionTreeRegressor(max_depth=8,random_state=42,min_samples_split=25,max_leaf_nodes=50, max_features=0.6)
param_dectree= {'max_depth':[4,5,6,7,8,9] }

dectree = model_fit(dectree, xtrain, ytrain_casual, param_dectree, False)
dectree = model_fit(dectree, xtrain, ytrain_registered, param_dectree, False)

ypred_dectree_count = Test_Set_Report("Decision Tree",dectree)

result_dectree = pd.concat([ytest_count,pd.Series(ypred_dectree_count)], axis=1)




''' RANDOM FOREST REGRESSOR '''
ranfor = RandomForestRegressor(n_estimators=60,criterion='mse',max_features=0.5,max_leaf_nodes=30, max_depth=6, min_samples_split=20,random_state=42)
param_ranfor= {'max_leaf_nodes':[10,20,30,40,50],'min_samples_split':[20,30,40,50]}

ranfor = model_fit(ranfor, xtrain, ytrain_casual, param_ranfor, False)
ranfor = model_fit(ranfor, xtrain, ytrain_registered, param_ranfor, False)

ypred_ranfor_count = Test_Set_Report("Random Forest",ranfor)

result_ranfor = pd.concat([ytest_count,pd.Series(ypred_ranfor_count)], axis=1)




''' XGBOOST REGRESSOR '''

xgbc= XGBRegressor(n_estimators=300, min_child_weight=2, max_depth=4, colsample_bylevel=0.5, gamma=0.1, reg_alpha=1, subsample=0.8, learning_rate=0.05, random_state=42)

param_xgb={'min_child_weight':[1,2,3,4,5]}
xgbc=model_fit(xgbc, xtrain, ytrain_casual, param_xgb, False)
xgbc=model_fit(xgbc, xtrain, ytrain_registered, param_xgb, False)

ypred_xgb_count = Test_Set_Report("XGBoost",xgbc)

result_xgb = pd.concat([ytest_count,pd.Series(ypred_xgb_count)], axis=1)

# xgb.cv is used to get the actual number of n_estimators required based on the learning rate, 
# it uses early_stopping_rounds to get the optimal value
xdtrain = xgb.DMatrix(xtrain,label=ytrain_casual)
cvresult_xgb=xgb.cv(xgbc.get_xgb_params(), xdtrain,nfold=5,num_boost_round=5000,metrics='rmse',early_stopping_rounds=50)

bestpred_xgb = print_feature_importance(xgbc)




''' Storing the predicted results along with the actual prediction for each phone number in a csv file'''

FinalResult_Python= FinalResult_Python.reset_index()
FinalResult_Python.drop(columns=['index'],axis=1,inplace=True)
FinalResult_Python = pd.concat([FinalResult_Python,result_xgb[0]], axis=1)
FinalResult_Python.rename(columns={0:'Predicted Count'}, inplace=True)
FinalResult_Python.to_csv("PredictedRentalCount_Python.csv", index=False)



















