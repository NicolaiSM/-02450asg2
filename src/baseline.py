import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
from data_handler import get_data
from sklearn.metrics import mean_squared_error, r2_score


def baseline(X, y, N, M, attributeNames, CV):

    errors = []

    for train_index, test_index in CV.split(X,y):  
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]

        pred = np.mean(y_train)
        errors.append(np.power(pred-y_test,2).sum().astype(float)/y_test.shape[0])

    return errors
        
    