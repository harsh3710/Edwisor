## libraries used
#randomForest, rpart, glmnet, lm, xgboost, data.table, mlr, ggplot2, gridextra

# set working directory
setwd("Downloads/Data Science Problem sets/Bike_Renting")

# importing test and train data into dataframes by and removing whitespaces
dataset = read.csv("day.csv", header = T, strip.white= T )
dataset_copy = data.frame(dataset)

# removing dteday and instant variable from the dataset
dataset = within(dataset, rm(dteday,instant))


#convert discreet variables  to categorical:
cat_features = c('season','yr','mnth','holiday','workingday','weekday','weathersit')
for(predictor in cat_features)
{
  dataset[[predictor]]=as.factor(dataset[[predictor]])
}


# mapping numerical categories to labels for better representation
levels(dataset$season) = c('spring','summer','fall','winter')
levels(dataset$weathersit) = c('clear.partly_cloudy','mist.cloudy','light_snow.rain','heavy_rain.snow')


# getting the descriptive statistics of the dataset
summary(dataset)


# histograms and boxplots to observe distribution for target variable(casual,registered,count)
for (feature in colnames(dataset)){
  if(class(dataset[[feature]])=='integer' ){
    gg1=ggplot(dataset,aes(x=dataset[[feature]]))+geom_histogram()+labs(x=feature)
    print(gg1)
    gg2= ggplot(dataset,aes(y=dataset[[feature]]))+geom_boxplot(fill="#4271AE")+labs(y=feature)
    print(gg2)
  }
}

# histograms to observe distribution for all numeric predictors
for (feature in colnames(dataset)){
  if(class(dataset[[feature]])=='numeric' ){
    gg=ggplot(dataset,aes(x=dataset[[feature]]))+geom_histogram()+labs(x=feature)
    print(gg)
  }
}


# plotting scatterplos to observe relation between continous variables
for (feature in colnames(dataset)){
  if(class(dataset[[feature]])=='numeric' ){
    gg1=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$casual))+geom_point()+labs(y='casual',x=feature)
    gg2=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$registered))+geom_point()+labs(y='registered',x=feature)
    gg3=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$cnt))+geom_point()+labs(y='count',x=feature)
    grid.arrange(gg1,gg2,gg3,ncol=1)
  }
}


# plotting boxplots for casual and registerd users for categorical variables
for (feature in colnames(dataset)){
  if(class(dataset[[feature]])=='factor' ){
    gg1=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$casual))+geom_boxplot(fill="#4271AE")+labs(y='casual',x=feature)
    gg2=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$registered))+geom_boxplot(fill="#4271AE")+labs(y='registered',x=feature)
    gg3=ggplot(dataset,aes(x=dataset[[feature]],y=dataset$cnt))+geom_boxplot(fill="#4271AE")+labs(y='count',x=feature)
    grid.arrange(gg1,gg2,gg3,ncol=1)
  }
}

# finding the count of missing values 
missing_values = data.frame(c(sapply(dataset, function(x)  sum(is.na(x)))))
colnames(missing_values) = 'Count'


# correlation matrix of all numeric predictors and target variable                        
dataset_corr = data.frame(lapply(dataset, function(x) {
  if(is.integer(x)) as.numeric(x) else  x}))
correlation_matrix = cor(dataset_corr[sapply(dataset_corr, is.numeric)], method = 'pearson')



# one hot encoding the below features
dummy_var = c('season','weathersit','mnth','weekday')
for(predictor in dummy_var){
for(unique_value in unique(dataset[[predictor]])){
  dataset[paste(predictor, unique_value, sep = ".")] = ifelse(dataset[[predictor]] == unique_value, 1, 0)
}}
dataset = within(dataset,rm(season,weathersit,mnth,weekday)) 


for(predictor in c('yr','holiday','workingday'))
{
  dataset[[predictor]]=as.integer(dataset[[predictor]])
}

# dropping temp feature and keeping atemp since both are highly correlated
dataset = within(dataset,rm(temp))

# dropping observation containing humidity=0 which is next to impossible
temp_data = dataset[dataset$hum!=0,]

# dropping observations containing outliers for casual variable for better accuracy
temp_data = temp_data[(temp_data$casual-mean(temp_data$casual))<=(3*sd(temp_data$casual)),]
dataset_copy = dataset_copy[(dataset_copy$casual-mean(dataset_copy$casual))<=(3*sd(dataset_copy$casual)),]


# splitting dataset into train and test  
smp_size = floor(0.9 * nrow(temp_data))
set.seed(42)
train_ind = sample(seq_len(nrow(temp_data)), size = smp_size)

train = temp_data[train_ind, ]
test = temp_data[-train_ind, ]  
FinalResult_R <- cbind(NA, NA)
FinalResult_R <- cbind(dataset_copy[-train_ind,])

ytest_count = data.frame(actual_count=test$cnt)
train_casual = within(train,rm(registered,cnt))
test_casual = within(test,rm(registered,cnt))

train_registered = within(train,rm(casual,cnt))
test_registered = within(test,rm(casual,cnt))


# creating training and testing tasks for casual users
traintask_casual = makeRegrTask(data= train_casual , target= 'casual' )
testtask_casual = makeRegrTask(data= test_casual , target= 'casual' )

# creating training and testing tasks for registered users
traintask_registered = makeRegrTask(data= train_registered , target= 'registered' )
testtask_registered = makeRegrTask(data= test_registered , target= 'registered' )


# generic function for tuning hyperparameters using gridsearch
model_tuning <- function(algoname, algo, params_tune, traintask ){
  set.seed(42)
  gridsearch = tuneParams(algo , resampling = makeResampleDesc("CV",iters = 5L, predict = "both"),
                          task= traintask, par.set = params_tune,
                          measures = list(rmse,setAggregation(rmse, train.mean)) ,
                          control = makeTuneControlGrid()
  )
  
  cat("\nGrid Search Report-",algoname,"\n")
  cat("Best params:\n")
  print(as.matrix(gridsearch$x))
  
  cat("\nEvaluation metrics for best model:Train Fold\n")
  cat("RMSE:",gridsearch$y[2],"\n")
  
  cat("\nEvaluation metrics for best model:Test Fold\n")
  cat("RMSE:",gridsearch$y[1],"\n")
  
  return(gridsearch$x)
}


# generic function to display cross validation scores for train dataset
cross_validation <- function(algoname, algo, traintask){
  set.seed(42)
  r=resample(algo, traintask, makeResampleDesc("CV",iters=5, predict="both"), 
             measures = list(rmse,setAggregation(rmse, train.mean)) )  
  
  cat("\nResampling Report-",algoname,"\n")
  cat("\nEvaluation metrics for best model:Train Fold\n")
  cat("RMSE:",r$aggr[2],"\n")
  
  cat("\nEvaluation metrics for best model:Test Fold\n")
  cat("RMSE:",r$aggr[1],"\n")
}


# generic function to evaluate the test set using optimal parameters
Test_Set_Report <- function (algoname, algo){
  
  model_fit = train(algo, traintask_casual)
  ypred_casual = predict(model_fit , testtask_casual)
  
  model_fit = train(algo, traintask_registered)
  ypred_registered = predict(model_fit , testtask_registered)
  
  ypred_count = data.frame(pred_count=ypred_casual$data$response + ypred_registered$data$response)
  
  cat("\nTest Set Report-",algoname,"\n")
  cat("\nEvaluation metrics:\n")
  cat("RMSE :",measureRMSE(ytest_count$actual_count,ypred_count$pred_count),"\n")
  cat("RMSLE :",measureRMSLE(ytest_count$actual_count,ypred_count$pred_count),"\n")
  
  df = cbind(ytest_count,ypred_count)
  gg=ggplot(df, aes(x=actual_count, y=pred_count)) + geom_point()
  print(gg)
  return(df)
}


# function to generate feature importance graph
featureImportance <- function(algoname, algo, traintask){
  ftimp = data.frame(t((getFeatureImportance(train(algo, traintask)))$res))
  setDT(ftimp, keep.rownames = TRUE)[]
  colnames(ftimp)= c('Feature','Importance')
  ftimp = ftimp[order(ftimp$Importance, decreasing = T),]
  ftimp = ftimp[1:15,]
  ftimp = ftimp[order(ftimp$Importance, decreasing = F),]
  level_order = ftimp$Feature
  gg = ggplot(ftimp, aes(x=factor(Feature, level = level_order),y=Importance))+geom_col()+labs(title=algoname, x='Feature')+coord_flip()
  print(gg)
  return(ftimp$Feature)
}


#### LINEAR REGRESSION ####
linreg_model = makeLearner("regr.lm", predict.type='response' )

# cross validation report
cross_validation("Linear Regression", linreg_model, traintask_casual )
cross_validation("Linear Regression", linreg_model, traintask_registered )

# evaluating the test set performance
ypred_count_linreg = Test_Set_Report("Linear Regression", linreg_model)



#### REGULARIZED LINEAR REGRESSION - RIDGE ####
ridge_model = makeLearner("regr.glmnet", predict.type='response' , par.vals = list(lambda=0.5,alpha=0))

#Search for hyperparameters
params_ridge = makeParamSet(
  makeDiscreteParam("lambda",values=c(10))
)

# perform grid search to get optimal parameters
bestparams_ridge_casual = model_tuning("Ridge Regression", ridge_model, params_ridge, traintask_casual)
bestparams_ridge_registered = model_tuning("Ridge Regression", ridge_model, params_ridge, traintask_registered)

# cross validation report using optimal parameters
ridge_model = setHyperPars(ridge_model , par.vals = bestparams_ridge_casual)
cross_validation("Ridge Regression", ridge_model, traintask_casual)
cross_validation("Ridge Regression", ridge_model, traintask_registered)

# evaluating the test set performance
ypred_count_ridge = Test_Set_Report("Ridge Regression", ridge_model)


#### REGULARIZED LINEAR REGRESSION - LASSO ####
lasso_model = makeLearner("regr.glmnet", predict.type='response' , par.vals = list(lambda=2,alpha=1))

#Search for hyperparameters
params_lasso = makeParamSet(
  makeDiscreteParam("lambda",values=c(0.1,0.2,0.5,1,2,3,5,6))
)

# perform grid search to get optimal parameters
bestparams_lasso_casual = model_tuning("Lasso Regression", lasso_model, params_lasso, traintask_casual)
bestparams_lasso_registered = model_tuning("Lasso Regression", lasso_model, params_lasso, traintask_registered)

# cross validation report using optimal parameters
lasso_model = setHyperPars(lasso_model , par.vals = bestparams_lasso_casual)
cross_validation("Lasso Regression", lasso_model, traintask_casual)
cross_validation("Lasso Regression", lasso_model, traintask_registered)

# evaluating the test set performance
ypred_count_ridge = Test_Set_Report("Ridge Regression", ridge_model)


#### DECISION TREE ####
# creating a learner
dectree_model = makeLearner("regr.rpart" , predict.type = "response", par.vals= list(minbucket=20,maxdepth=15, minsplit=5) )

#Search for hyperparameters
params_dectree = makeParamSet(
  makeDiscreteParam("minsplit",values=c(3,4,5,6)),
  makeDiscreteParam("minbucket", values=c(15,20,25)),
  makeDiscreteParam("maxdepth", values=c(11,12,14,15))
)

# perform grid search to get optimal parameters
bestparams_dectree_casual = model_tuning("Decision Tree", dectree_model, params_dectree, traintask_casual)
bestparams_dectree_registered = model_tuning("Decision Tree", dectree_model, params_dectree, traintask_registered)

# cross validation report using optimal parameters
dectree_model = setHyperPars(dectree_model , par.vals = bestparams_dectree_casual)
cross_validation("Decison Tree", dectree_model, traintask_casual)
cross_validation("Decison Tree", dectree_model, traintask_registered)

# plot feature importance for decision trees
bestpred_dectree_casual= featureImportance("Decision Tree",dectree_model, traintask_casual)
bestpred_dectree_registered= featureImportance("Decision Tree",dectree_model, traintask_registered)

# evaluating the test set performance
ypred_count_dectree = Test_Set_Report("Decision Tree", dectree_model)



#### RANDOM FOREST CLASSIFIER ####
ranfor_model = makeLearner("regr.randomForest" , predict.type = "response", par.vals= list(ntree= 60,mtry=12, nodesize= 20, maxnodes= 60, importance=T))

#Search for hyperparameters
params_ranfor = makeParamSet(
  #makeDiscreteParam("ntree",values=c(45,55,60,65)),
  #makeDiscreteParam('mtry', values=c(6,9,12,15)),
  makeDiscreteParam("nodesize", values=c(20,25,30)),
  makeDiscreteParam("maxnodes", values=c(50,55,60))
)

# perform grid search to get optimal parameters
bestparams_ranfor_casual = model_tuning("Random Forest", ranfor_model, params_ranfor, traintask_casual)
bestparams_ranfor_registered = model_tuning("Random Forest", ranfor_model, params_ranfor, traintask_registered)

# cross validation report using optimal parameters
ranfor_model = setHyperPars(ranfor_model , par.vals = bestparams_ranfor_casual)
cross_validation("Random Forest", ranfor_model, traintask_casual)
cross_validation("Random Forest", ranfor_model, traintask_registered)

# plot feature importance for random forest
bestpred_rf_casual= featureImportance("Random Forest",ranfor_model, traintask_casual)
bestpred_rf_registered= featureImportance("Random Forest",ranfor_model, traintask_registered)


# evaluating the test set performance
ypred_count_ranfor= Test_Set_Report("Random Forest", ranfor_model)




#### XGBOOST CLASSIFIER ####

#make learner with inital parameters
xgb_model <- makeLearner("regr.xgboost", predict.type = "response")
xgb_model$par.vals <- list(
  objective = "reg:linear",
  nrounds = 500,
  eta= 0.05,
  subsample= 0.9,
  colsample_bytree= 0.5,
  eval_metric="rmse",
  early_stopping_rounds=50,
  verbose=0,
  print_every_n = 25,
  max_depth= 4,
  min_child_weight = 2,
  gamma = 0.1,
  alpha = 1
)


#Search for hyperparameters
params_xgb = makeParamSet(
  makeDiscreteParam("max_depth",values=c(5)),
  makeDiscreteParam("gamma",values=c(0)),
  makeDiscreteParam('min_child_weight', values=c(1)),
  makeDiscreteParam("subsample", values=c(1)),
  makeDiscreteParam("colsample_bytree", values=c(0.8)),
  makeDiscreteParam("max_delta_step", values=c(0.32)),
  makeDiscreteParam("alpha", values=c(0.1,0.5,1)),
  makeDiscreteParam("lambda", values=c(0.1,0.5,1))
)


# perform grid search to get optimal parameters
bestparams_casual_xgb = model_tuning("XGBoost", xgb_model, params_xgb, traintask_casual)
bestparams_registered_xgb = model_tuning("XGBoost", xgb_model, params_xgb, traintask_registered)

# cross validation report using optimal parameters
xgb_model = setHyperPars(xgb_model , par.vals = bestparams_casual_xgb)
cross_validation("XGBoost", xgb_model, traintask_casual)
cross_validation("XGBoost", xgb_model, traintask_registered)

# plot feature importance for random forest
bestpred_xgb_casual = data.frame(featureImportance("XGBoost",xgb_model, traintask_casual))
bestpred_xgb_registered = data.frame(featureImportance("XGBoost",xgb_model, traintask_registered))


# evaluating the test set performance
ypred_count_xgb = Test_Set_Report("XGBoost", xgb_model)


### Storing the predicted rental count along with the actual count for each day in a csv file
FinalResult_R = cbind(FinalResult_R, ypred_count_xgb$pred_count)
colnames(FinalResult_R)[17]= c( 'Predicted_Count')
write.csv(FinalResult_R, file="PredictedRentalCount_R.csv", row.names = F)



