# -*- coding: utf-8 -*-
"""
@author: Ammad Hameed (s174297)
"""

# exercise 2.1.1
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import math
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot,rlr_validate
import neurolab as nl
from scipy import stats


from scipy.linalg import svd
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from scipy import stats

# Load xls sheet with data
doc = xlrd.open_workbook('flagData.xlsx').sheet_by_index(0)

# Extract attribute names (1st row, column 1 to 28)
attributeNames = doc.row_values(0, 1, 56)

del attributeNames[28]
del attributeNames[29-1]
del attributeNames[30-2]
del attributeNames[31-3]
del attributeNames[32-4]
del attributeNames[33-5]
del attributeNames[34-6]
del attributeNames[35-7]

np.random.seed(2) #2

# Preallocate memory, then extract excel data to matrix X
X1 = np.empty((194, 46))  
for i, col_id in enumerate(range(1, 29)):
    X1[:, i] = np.asarray(doc.col_values(col_id, 1, 195))
for i, col_id in enumerate(range(37, 55)):
    X1[:, i+28] = np.asarray(doc.col_values(col_id, 1, 195))
N, M = X1.shape

X = X1 - (np.ones((N, 1)) * X1.mean(axis=0))
X = X / X.std(axis=0)

y1 = np.asarray(doc.col_values(29,1,195))  

y = y1 - (np.ones((1)) * y1.mean(axis=0))
y = y / y.std(axis=0)

# To divide the one-of-K encoded columns with sqrt(noOfVariables) (standardizing)
def divideOneHot(index, noOfVariables):
    X[:,index] = X[:,index]*(1/math.sqrt(noOfVariables))
    return X

for i in range(6):
    divideOneHot(i,6)
    
for i in range(10):
    divideOneHot(i+8,10)
    
for i in range(8):    
    divideOneHot(i+18,8)


### FORWARD SELECTION / LR ###


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.zeros(K)*np.nan
Error_test = np.zeros(K)*np.nan
Error_train_fs = np.zeros(K)*np.nan
Error_test_fs = np.zeros(K)*np.nan
Error_train_nofeatures = np.zeros(K)*np.nan
Error_test_nofeatures = np.zeros(K)*np.nan
Lowest_error = np.zeros(K)*np.nan

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Lowest_error[k] = min(loss_record)
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        plt.figure(k)
        plt.subplot(1,2,1)
        plt.plot(range(1,len(loss_record)), loss_record[1:])
        plt.xlabel('Iteration')
        plt.ylabel('Squared error (crossvalidation)')    
        
        plt.subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        plt.clim(-1.5,0)
        plt.xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

plt.figure(k)
plt.subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
plt.clim(-1.5,0)
plt.xlabel('Crossvalidation fold')
plt.ylabel('Attribute')

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=np.argmin(Lowest_error)+1 # cross-validation fold to inspect
print(f)
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    y_est_LR= m.predict(X[:,ff])
    residual=y-y_est_LR
    
    plt.figure(k+1, figsize=(12,6))
    plt.title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       plt.subplot(2,np.ceil(len(ff)/2.0),i+1)
       plt.plot(X[:,ff[i]],residual,'.')
       plt.xlabel(attributeNames[ff[i]])
       plt.ylabel('residual error')
    
    
plt.show()

X_fs = np.zeros((194,10))
X_fs[:, 0] = X[:,2]
X_fs[:, 1] = X[:,6]
X_fs[:, 2] = X[:,7]
X_fs[:, 3] = X[:,8]
X_fs[:, 4] = X[:,24]
X_fs[:, 5] = X[:,27]
X_fs[:, 6] = X[:,38]
X_fs[:, 7] = X[:,42]
X_fs[:, 8] = X[:,43]
X_fs[:, 9] = X[:,44]


model = lm.LinearRegression()
model.fit(X_fs,y)
w = model.coef_
print('w: ',w)
y_est_LR = model.predict(X_fs)
residual = y_est_LR-y

plt.bar(range(0,K),Error_test)
plt.title('LR: Mean-square errors (MSE)');
print('LR: Root Mean-square error (RMSE): {0}'.format((math.sqrt(np.mean(Error_test)))))
print('Error Test:',Error_test)

# Display plots
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(y, y_est_LR, '.g')
plt.xlabel('No. of colors in flag (standardized, true)'); plt.ylabel('No. of colors in flag (standardized, estimated)')
plt.show()

### ANN ###
np.random.seed(2) #2

# Normalize data
X = stats.zscore(X1);
y = stats.zscore(y1);

# Parameters for neural network classifier
n_hidden_units = 3      # number of hidden units
hidden_units = np.zeros(5)
n_train = 2             # number of networks trained in each k-fold
learning_goal = 5     # stop criterion 1 (train mse to be reached)
max_epochs = 100         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates

# K-fold crossvalidation
K_outer = 5
K = 5
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
k=0

for outer_train_index, outer_test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K_outer))    
    
    # extract training and test set for current CV fold
    X_train_outer = X[outer_train_index,:]
    y_train_outer = y[outer_train_index]
    X_test_outer = X[outer_test_index,:]
    y_test_outer = y[outer_test_index]
    
    for train_index, test_index in CV.split(X_train_outer,y_train_outer):
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        best_train_error = np.inf
        for j in range(1,K):
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i+1,n_train))
                # Create randomly initialized network with 2 layers
                ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
                if i==0:
                    bestnet.append(ann)
                # train network
                train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
                if train_error[-1]<best_train_error:
                    bestnet[k]=ann
                    best_train_error = train_error[-1]
                    error_hist[range(len(train_error)),k] = train_error
                    n_hidden_units = j
                    
            print('Best train error: {0}...'.format(best_train_error))
            y_est_ANN = bestnet[k].sim(X_test).squeeze()
            errors[k] = np.power(y_est_ANN-y_test,2).sum().astype(float)/y_test.shape[0]
            hidden_units[k] = n_hidden_units
    for i in range(n_train):
            print('Training network {0}/{1}...'.format(i+1,n_train))
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            if i==0:
                bestnet.append(ann)
            # train network
            train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1]<best_train_error:
                bestnet[k]=ann
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k] = train_error
                
    print('Best train error: {0}...'.format(best_train_error))
    y_est_ANN = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est_ANN-y_test,2).sum().astype(float)/y_test.shape[0]
    k+=1       
print('Errors:',errors)
print('Hidden units:',hidden_units)


# Print the average least squares error
print('ANN: Root Mean-square error (RMSE): {0}'.format(math.sqrt(np.mean(errors))))

plt.bar(range(0,K),errors); plt.title('ANN: Mean-square errors (MSE)'); plt.show();

plt.plot(y_est_ANN); plt.plot(y_test); plt.title('est_y vs. test_y'); plt.legend(('y_est', 'y_test')); plt.show();
plt.plot((y_est_ANN-y_test)); plt.title('Prediction error (est_y-test_y)'); plt.show();

# Display plots
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(y_test, y_est_ANN, '.g')
plt.xlabel('No. of colors in flag (standardized, true)'); plt.ylabel('No. of colors in flag (standardized, estimated)')
plt.show()


### Baseline - Prediction Average ###
# K-fold crossvalidation
np.random.seed(2) #2

K = 5
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
base_errors = np.zeros(K)*np.nan

k=0
for base_train_index, base_test_index in CV.split(X,y):
    print('\nBaseline Crossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    base_X_train = X[base_train_index,:]
    base_y_train = y[base_train_index]
    base_X_test = X[base_test_index,:]
    base_y_test = y[base_test_index]

    base_y_est = np.mean(base_y_train)
    base_errors[k] = np.power(base_y_est-base_y_test,2).sum().astype(float)/y_test.shape[0]
    k+=1
    
plt.bar(range(0,K),base_errors)
plt.title('Pedicted Average: Mean-square errors (MSE)');
print('Predicted Average: Root Mean-square error (RMSE): {0}'.format((math.sqrt(np.mean(base_errors)))))
print('Predicted Average Error Test:',base_errors)
plt.show()

### Statistics ###
# ANN - LR #
z_ANN_LR = (Error_test-errors)
zb_ANN_LR = z_ANN_LR.mean()
nu = K-1
sig =  (z_ANN_LR-zb_ANN_LR).std()  / np.sqrt(K-1)
alpha = 0.05

zL_ANN_LR = zb_ANN_LR + sig * stats.t.ppf(alpha/2, nu);
zH_ANN_LR = zb_ANN_LR + sig * stats.t.ppf(1-alpha/2, nu);

if zL_ANN_LR <= 0 and zH_ANN_LR >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
print('Interval is: [',zL_ANN_LR,':',zH_ANN_LR,']')
    
# Boxplot to compare classifier error distributions
plt.figure()
plt.boxplot(np.concatenate((Error_test.reshape(-1,1), errors.reshape(-1,1)),axis=1))
plt.xlabel('Linear Regression   vs.   ANN')
plt.ylabel('Cross-validation error [%]')

plt.show()

# ANN - PA #
z_ANN_PA = (errors-base_errors)
zb_ANN_PA = z_ANN_PA.mean()
nu = K-1
sig =  (z_ANN_PA-zb_ANN_PA).std()  / np.sqrt(K-1)
alpha = 0.05

zL_ANN_PA = zb_ANN_PA + sig * stats.t.ppf(alpha/2, nu);
zH_ANN_PA = zb_ANN_PA + sig * stats.t.ppf(1-alpha/2, nu);

if zL_ANN_PA <= 0 and zH_ANN_PA >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
print('Interval is: [',zL_ANN_PA,':',zH_ANN_PA,']')
    
# Boxplot to compare classifier error distributions
plt.figure()
plt.boxplot(np.concatenate((errors.reshape(-1,1), base_errors.reshape(-1,1)),axis=1))
plt.xlabel('ANN   vs.   Predicted Average')
plt.ylabel('Cross-validation error [%]')

plt.show()

# PA - LR #
z_PA_LR = (base_errors-Error_test)
zb_PA_LR = z_PA_LR.mean()
nu = K-1
sig =  (z_PA_LR-zb_PA_LR).std()  / np.sqrt(K-1)
alpha = 0.05

zL_PA_LR = zb_PA_LR + sig * stats.t.ppf(alpha/2, nu);
zH_PA_LR = zb_PA_LR + sig * stats.t.ppf(1-alpha/2, nu);

if zL_PA_LR <= 0 and zH_PA_LR >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
print('Interval is: [',zL_PA_LR,':',zH_PA_LR,']')
    
# Boxplot to compare classifier error distributions
plt.figure()
plt.boxplot(np.concatenate((base_errors.reshape(-1,1), Error_test.reshape(-1,1)),axis=1))
plt.xlabel('Predicted Average   vs.   Linear Regression')
plt.ylabel('Cross-validation error [%]')

plt.show()