from data_handler import get_data
from sklearn import model_selection
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt


def rlr_validate(data_train_outer, target_train_outer, data_test_outer, target_test_outer, lambdas, M, cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''

    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    
    for index in range(len(data_train_outer)):

        X_train = data_train_outer[index]
        y_train = target_train_outer[index]
        X_test = data_test_outer[index]
        y_test = target_test_outer[index]
    
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
    

def linearmodel(N, M, attributeNames, lambdas, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer, K):

    MSE_Error_train = np.empty((K,1))
    MSE_Error_test = np.empty((K,1))
    MSE_Error_train_rlr = np.empty((K,1))
    MSE_Error_test_rlr = np.empty((K,1))
    MSE_Error_train_nofeatures = np.empty((K,1))
    MSE_Error_test_nofeatures = np.empty((K,1))


    GE_Error_train = np.empty((K,1))
    GE_Error_test = np.empty((K,1))
    GE_Error_train_rlr = np.empty((K,1))
    GE_Error_test_rlr = np.empty((K,1))
    GE_Error_train_nofeatures = np.empty((K,1))
    GE_Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.empty((M,K))
    w_noreg = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    lambdas_opt = np.empty((K,1))
    target = []
    pred = []

    k=0

    for index in range(len(data_train)):

        X_train = data_train[index]
        y_train = target_train[index]
        X_test = data_test[index]
        y_test = target_test[index]
        

        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(data_train_outer[index], target_train_outer[index], data_test_outer[index], target_test_outer[index], lambdas, M)
        lambdas_opt[k] = opt_lambda

        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Compute mean squared error without using the input data at all
        GE_Error_train_nofeatures[k] = np.power(y_train-y_train.mean(), 2).sum(axis=0)/y_train.shape[0]
        GE_Error_test_nofeatures[k] = np.power(y_test-y_test.mean(), 2).sum(axis=0)/y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        GE_Error_train_rlr[k] = np.power(y_train-X_train @ w_rlr[:,k], 2).sum(axis=0)/y_train.shape[0]
        GE_Error_test_rlr[k] = np.power(y_test-X_test @ w_rlr[:,k], 2).sum(axis=0)/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        GE_Error_train[k] = np.power(y_train-X_train @ w_noreg[:,k], 2).sum(axis=0)/y_train.shape[0]
        GE_Error_test[k] = np.power(y_test-X_test @ w_noreg[:,k], 2).sum(axis=0)/y_test.shape[0]


        # Compute mean squared error without using the input data at all
        MSE_Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        MSE_Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

        # Compute mean squared error with regularization with optimal lambda
        MSE_Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        MSE_Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        # Compute mean squared error without regularization
        MSE_Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        MSE_Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]


        pred.append(X_test @ w_rlr[:,k]) 
        target.append(y_test)

        k+=1



    lr_GE_error = {
        "GE_Error_train":GE_Error_train,
        "GE_Error_test":GE_Error_test,
        "GE_Error_train_rlr":GE_Error_train_rlr,
        "GE_Error_test_rlr":GE_Error_test_rlr,
        "GE_Error_train_nofeatures":GE_Error_train_nofeatures,
        "GE_Error_test_nofeatures":GE_Error_test_nofeatures,

    }

    lr_MSE_error = {
        "MSE_Error_train":MSE_Error_train,
        "MSE_Error_test":MSE_Error_test,
        "MSE_Error_train_rlr":MSE_Error_train_rlr,
        "MSE_Error_test_rlr":MSE_Error_test_rlr,
        "MSE_Error_train_nofeatures":MSE_Error_train_nofeatures,
        "MSE_Error_test_nofeatures":MSE_Error_test_nofeatures,

    }

    lr_cross_val_last = {
        "opt_val_err":opt_val_err, 
        "opt_lambda":opt_lambda, 
        "mean_w_vs_lambda":mean_w_vs_lambda, 
        "train_err_vs_lambda":train_err_vs_lambda, 
        "test_err_vs_lambda":test_err_vs_lambda
    }

    lr_rest = {
        "w_rlr":w_rlr,
        "w_noreg":w_noreg,
        "mu":mu,
        "sigma":sigma,
        "lambdas_opt":lambdas_opt,
        "target":target,
        "pred":pred,
    }

    print("----------linreg_done-----------")

    return lr_GE_error, lr_MSE_error, lr_cross_val_last, lr_rest



