from data_handler import get_data
from ann_reg import ANN_trainer
from baseline import baseline
from linearmodel import linearmodel
import numpy as np
import matplotlib.pyplot as plt

def cross_validation_func(data, target, N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer):


    
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






K=10

target = "colours"

drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language", "religion"]


data, target, N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer = get_data(K, onehot_classes, drop_columns, target)


lr_GE_error, lr_MSE_error, lr_cross_val_last, lr_rest, baseline_errors, baseline_preds, baseline_y_tests, ann_n_hidden, ann_error, ann_test_predict, ann_test_true, ann_train_loss = cross_validation_func(data, target, N, M, attributeNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer)



lambdas = np.power(10.,range(-5,9))


"""
plt.figure(10, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,lr_cross_val_last["mean_w_vs_lambda"].T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(lr_cross_val_last["opt_lambda"])))
plt.loglog(lambdas,lr_cross_val_last["train_err_vs_lambda"].T,'b.-',lambdas,lr_cross_val_last["test_err_vs_lambda"].T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()
"""


opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(data_train, target_train, data_test, target_test, lambdas, M)

plt.figure(10, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
plt.show()

# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()
plt.show()



print([round(a,3) for a in baseline_errors]) 
print([round(a,3) for a in ann_n_hidden])
print([round(a,3) for a in ann_error])
print([round(a,3) for a in lr_rest["lambdas_opt"].squeeze()]) 
print([round(a,3) for a in lr_GE_error["GE_Error_test_rlr"].squeeze()])



index = 7

legend = ["ANN - E: " + str(round(ann_error[index], 3)) ,
           "Linear Reg - E: " + str(round(lr_GE_error["GE_Error_test_rlr"].squeeze()[index], 3)),
           'Baseline - E: ' + str(round(baseline_errors[index], 3)),
           'test data']


x = range(1,len(baseline_y_tests[index])+1)
plt.plot(x, ann_test_predict[index])
plt.plot(x, lr_rest["pred"][index])
plt.plot(x, [baseline_preds[index]]*len(baseline_y_tests[index]))
plt.plot(x, baseline_y_tests[index])
plt.title("Comparison between predicted result and true solution")
plt.xlabel('Number of colors (standardized)')
plt.ylabel('Observation number')
plt.xlim(1,len(baseline_y_tests[index]))
plt.legend(legend)
plt.grid()
plt.show()



def ttest_twomodels(y_true, yhatA, yhatB, alpha=0.05, loss_norm_p=1):
    zA = np.abs(y_true - yhatA) ** loss_norm_p
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zB = np.abs(y_true - yhatB) ** loss_norm_p

    z = zA - zB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return np.mean(z), CI, p





baseline = []
for i in range(len(baseline_y_tests)):
    baseline.append([baseline_preds[i]]*len(baseline_y_tests[i]))

y_true = np.concatenate(baseline_y_tests)
baseline = np.concatenate(baseline)
ann = np.concatenate(ann_test_predict)
lr = np.concatenate(lr_rest["pred"])

bl_vs_ann = ttest_twomodels(y_true, ann, baseline )
bl_vs_lr = ttest_twomodels(y_true, lr, baseline)
lr_vs_ann = ttest_twomodels(y_true, ann, lr)


print(bl_vs_ann)
print(bl_vs_lr)
print(lr_vs_ann)


legend = ["ANN",
           "Linear Reg",
           "Baseline",
           'test data']


x = range(1,len(y_true)+1)
plt.plot(x, ann)
plt.plot(x, lr)
plt.plot(x, baseline)
plt.plot(x, y_true)
plt.title("Comparison between predicted result and true solution")
plt.xlabel('Number of colors (standardized)')
plt.ylabel('Observation number')
plt.xlim(1,len(x))
plt.legend(legend)
plt.grid()
plt.show()



"""
print([round(a,3) for a in baseline_errors]) 
print([round(a,3) for a in ann_n_hidden])
print([round(a,3) for a in ann_error])
print([round(a,3) for a in lr_rest["lambdas_opt"].squeeze()]) 
print([round(a,3) for a in lr_GE_error["GE_Error_test_rlr"].squeeze()])
"""


legend = ["ANN",
           "Linear Reg",
           'Baseline',]


x = range(1,len(ann_error)+1)
plt.plot(x, ann_error)
plt.plot(x, lr_GE_error["GE_Error_test_rlr"].squeeze())
plt.plot(x, baseline_errors)
plt.title("Generalisation error for 10 fold crossvalidation")
plt.xlabel('K-fold number')
plt.ylabel('Generalisation error')
plt.xlim(1,len(x))
plt.legend(legend)
plt.grid()



attributeNames2= ['area', 'population', 'bars', 'stripes', 'circles', 'crosses',
       'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon',
       'animate', 'text', 'landmass_1', 'landmass_2',
       'landmass_3', 'landmass_4', 'landmass_5', 'landmass_6', 'zone_1',
       'zone_2', 'zone_3', 'zone_4', 'language_1', 'language_2', 'language_3',
       'language_4', 'language_5', 'language_6', 'language_7', 'language_8',
       'language_9', 'language_10', 'religion_0', 'religion_1', 'religion_2',
       'religion_3', 'religion_4', 'religion_5', 'religion_6', 'religion_7']

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames2[m], np.round(lr_rest["w_rlr"][m,-1],2)))



































