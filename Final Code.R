
# R-code for the PROJECT
# 
# IE-7280
# Instructor: Shahin Shahrampour
# 
# Group Members:
# -Yihyun Kim
# -Souri Sasanfar
# -Alican Yilmaz
# -Negin Maddah


# Libraries
library("data.table")
library("readxl")
library(corrplot)
library(caret)
library("elasticnet")

# Reading Data

dataset <- read_excel("./train-1.xlsx")
data.table(dataset)

# Cross Validation

ctrl_2=trainControl(method='cv',number=5) # cv 5-fold
ctrl_3=trainControl(method='LOOCV') # LEAVE ONE OUT CV

# Model Training for the Selected Models

set.seed(425)

## Linear Regression with Stepwise Selection
fit=train(Y1~., data=dataset, method="lmStepAIC", metric="RMSE",trControl = ctrl_2) 
fit$results # This displays the performance metrics
summary(fit) # This displays the statistics of the coefficients
coef(fit$finalModel, 7) # This displays the coefficients of the Final model explicitly


set.seed(425)

## Polynomial Regression 

dataset_t <- data.table(dataset)
dataset_poly<-dataset_t[, ':='(Z11=X1*X1,Z22=X2*X2,Z33=X3*X3,Z44=X4*X4,Z55=X5*X5,Z66=X6*X6,Z77=X7*X7,Z88=X8*X8,Z111=X1*X1*X1,Z222=X2*X2*X2,Z333=X3*X3*X3,Z444=X4*X4*X4,Z555=X5*X5*X5,Z666=X6*X6*X6,Z777=X7*X7*X7,Z888=X8*X8*X8)]

fit=train(Y1~., data=dataset_poly, method="lmStepAIC", metric="RMSE",trControl = ctrl_2) 
fit$results  # This displays the performance metrics
summary(fit) # This displays the statistics of the coefficients


# Code for the rest of the models we trained (including the gbm and brnn)

## Linear Regression
fit=train(Y1~.-X2-X5, data=dataset, method="lm", metric="RMSE",trControl = ctrl_2)
fit$results
summary(fit)
fit$finalModel

## Linear Regression with Backwards Selection
grid <- expand.grid(nvmax=c(2,3,4,5,6))
fit=train(Y1~., data=dataset, method="leapBackward",tuneGrid=grid, metric="RMSE",trControl = ctrl_2)
fit$results
summary(fit)
coef(fit$finalModel, 7)

## Linear Regression with Forwards Selection
grid <- expand.grid(nvmax=c(2,3,4,5,6))
fit=train(Y1~., data=dataset, method="leapForward",tuneGrid=grid, metric="RMSE",trControl = ctrl_2)
fit$results
summary(fit)
coef(fit$finalModel, 7)

## Linear Regression with Stepwise Selection

fit=train(Y1~., data=dataset, method="leapSeq", metric="RMSE",tuneLength=5,trControl = ctrl_2) # adjust nvmax parameter
fit$results
summary(fit)
coef(fit$finalModel, 7)

## Ridge regression

## LASSO
fit=train(Y1~., data=dataset, method="lasso", metric="RMSE",tuneLength=5,trControl = ctrl_2) # lasso deals with multicollinearity
fit$results

## Gradient boosting machine

fit=train(Y1~., data=dataset, method="gbm", metric="RMSE",tuneLength=5,trControl = ctrl_2)
fit$results
summary(fit) # Shows the importance scores of the variables

## Bayesian Regularized Neural Networks

fit=train(Y1~., data=dataset, method="brnn", metric="RMSE",tuneLength=10,trControl = ctrl_2)
fit$resultsx
summary(fit)


# Leave one out cross validation

## Stepwise linear regression
fit=train(Y1~., data=dataset, method="lmStepAIC", metric="RMSE",trControl = ctrl_3) 
fit$results # This displays the performance metrics

## Polynomial regression
fit=train(Y1~., data=dataset_poly, method="lmStepAIC", metric="RMSE",trControl = ctrl_3)
fit$results


