from data_handler import get_data
from ann_reg import ANN_trainer
from baseline import baseline
from linearmodel import linearmodel
import numpy as np

def cross_validation_func():

    K=10

    target = "colours"

    drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"]

    onehot_classes = ["landmass", "zone", "language", "religion"]

    data, target, N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer = get_data(K, onehot_classes, drop_columns, target)

    baseline_errors, baseline_preds, baseline_y_tests= baseline(N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer)

    lambdas = np.power(10.,range(-5,9))


    lr_GE_error, lr_MSE_error, lr_cross_val_last, lr_rest = linearmodel(N, M, attributeNames, lambdas, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer, K)


    n_hiddens = [1, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    ann = ANN_trainer()

    ann.run( N, M, attributeNames, n_hiddens, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer)


    return lr_GE_error, lr_MSE_error, lr_cross_val_last, lr_rest, baseline_errors, baseline_preds, baseline_y_tests, ann.n_hidden, ann.error, ann.test_predict, ann.test_true, ann.train_loss



lr_GE_error, lr_MSE_error, lr_cross_val_last, lr_rest, baseline_errors, baseline_preds, baseline_y_tests, ann_n_hidden, ann_error, ann_test_predict, ann_test_true, ann_train_loss = cross_validation_func()

dic = {
"ann_n_hidden":ann_n_hidden, 
"ann_error":ann_error, 
"ann_test_predict":ann_test_predict, 
"ann_test_true":ann_test_true, 
"ann_train_loss":ann_train_loss, 
"baseline_errors":baseline_errors, 
"baseline_preds":baseline_preds, 
"baseline_y_tests":baseline_y_tests}


print(10*"-")
print(dic)
print(10*"-")
print(lr_GE_error)
print(10*"-")
print(lr_MSE_error)
print(10*"-")
print(lr_cross_val_last)
print(10*"-")
print(lr_rest)
print(10*"-")







