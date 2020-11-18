import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
from data_handler import get_data
from sklearn.metrics import mean_squared_error, r2_score


def baseline(N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer):

    errors = []
    preds = []
    y_tests = []

    for index in range(len(data_train)):  
        
        # extract training and test set for current CV fold
        X_train = data_train[index]
        y_train = target_train[index]
        X_test = data_test[index]
        y_test = target_test[index]

        pred = np.mean(y_train)
        errors.append(np.power(pred-y_test,2).sum().astype(float)/y_test.shape[0])
        preds.append(pred)
        y_tests.append(y_test)

    return errors, preds, y_tests

