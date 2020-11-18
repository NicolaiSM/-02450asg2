from data_handler import get_data
import numpy as np
import pandas as pd
from scipy import stats
from toolbox_02450 import mcnemar
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# from __main__ import K, onehot_classes, drop_columns


K = 10

drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red",
                "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language"]

X, yDontCare, y, N, M, attributeNames, classNames, CV = get_data(K, onehot_classes, drop_columns)

X = X - (np.ones((N, 1)) * X.mean(axis=0))
X = X / X.std(axis=0)
X = np.square(X)
N = len(y)
M = len(attributeNames)

# Three models
yhat1 = []
yhat2 = []
yhat3 = []
y_true = []

out = pd.DataFrame()


def cross_validation(z):
    k = 0
    best_lambda = []
    best_error = []
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        # Exercise 8_1_2

        if z == 1:
            mu = np.mean(X_train, 0)
            sigma = np.std(X_train, 0)
            X_train = (X_train - mu) / sigma
            X_test = (X_test - mu) / sigma

        lambda_interval = np.logspace(-8, 2, 15)
        # print(lambda_interval)
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))

        coefficent_list = np.array([])

        for i in range(len(lambda_interval)):
            if z == 1:
                mdl = LogisticRegression(solver='liblinear', penalty='l2', C=1 / lambda_interval[i],
                                         max_iter=1000000000)  # change to saga

            if z == 2:
                mdl = tree.DecisionTreeClassifier(ccp_alpha=1 / lambda_interval[i])

            mdl.fit(X_train, y_train)
            # coefficent_list = mdl.coef_

            if k == 8 and z == 1:
                coefficent_list = np.hstack((coefficent_list, mdl.coef_[1]))

            y_train_estimate = mdl.predict(X_train).T
            y_test_estimate = mdl.predict(X_test).T
            # print(y_train_estimate)
            # print(y_test_estimate)

            train_error_rate[i] = np.sum(y_train_estimate != y_train) / len(y_train)
            test_error_rate[i] = np.sum(y_test_estimate != y_test) / len(y_test)
            # print(test_error_rate)

        index = np.argmin(test_error_rate)
        best_upsidedown_y = lambda_interval[index]

        # print(coefficent_list)

        # print(best_upsidedown_y)
        # mdl = LogisticRegression(C=1 / best_upsidedown_y)
        # mdl.fit(X_train, y_train)

        if z == 1:
            mdl = LogisticRegression(C=1 / best_upsidedown_y)
        if z == 2:
            mdl = tree.DecisionTreeClassifier(ccp_alpha=1 / best_upsidedown_y)

        mdl.fit(X_train, y_train)

        y_test_estimate = mdl.predict(X_test).T
        #print(y_test_estimate)

        if z == 1:
            yhat1.extend(y_test_estimate)
            print(yhat1)
        if z == 2:
            yhat2.extend(y_test_estimate)

        Error_test_model = np.sum(y_test_estimate != y_test) / len(y_test)
        print('Cross validation fold {0}/{1}'.format(k + 1, K))

        accuracy = 100 - 100 * np.mean(y_test_estimate != y_test)

        best_lambda.append(np.round(best_upsidedown_y, 4))
        best_error.append(np.round(Error_test_model, 4))
        print('Best lamda value: ', best_upsidedown_y)
        print('Test error: ', np.round(Error_test_model, 4))

        cm = confusion_matrix(y_test, y_test_estimate)
        plt.figure(2)
        plt.imshow(cm, cmap='binary', interpolation='None')
        plt.colorbar()
        plt.xticks(range(len(classNames)))
        plt.yticks(range(len(classNames)))
        plt.xlabel('Predicted class')
        plt.ylabel('Actual class')
        plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, Error_test_model))
        plt.show()

        k += 1

        if k == K - 1 and z == 1:
            # coefficent_list2 = np.empty(())
            print(len(coefficent_list))
            print(coefficent_list.shape)
            list = [coefficent_list[j:j + 34] for j in range(0, len(coefficent_list), 34)]

            plt.figure(k, figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.semilogx(lambda_interval, list, '.-')  # Don't plot the bias term
            plt.xlabel('Regularization factor')
            plt.ylabel('Mean Coefficient Values')
            plt.grid()
            # You can choose to display the legend, but it's omitted for a cleaner
            # plot, since there are many attributes
            # legend(attributeNames[1:], loc='best')

            # print(train_error_rate.T)
            # print(test_error_rate.T)

            plt.subplot(1, 2, 2)
            plt.title('Optimal lambda: 1e{0}'.format(np.log10(best_upsidedown_y)))
            plt.loglog(lambda_interval, train_error_rate.T, 'b.-', lambda_interval, test_error_rate.T, 'r.-')
            plt.xlabel('Regularization factor')
            plt.ylabel('Squared error (crossvalidation)')
            plt.legend(['Train error', 'Validation error'])
            plt.grid()
            plt.show()

    return best_lambda, best_error


def logistic_regression():
    best_lambda, best_error = cross_validation(1)
    best_lambda = np.array(best_lambda).squeeze()
    out['l'] = best_lambda
    best_error = np.array(best_error).squeeze()
    out['log'] = best_error

    plt.title('Error rate over number of neighbours')
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], best_error)
    plt.xlabel('K-neighbours')
    plt.ylabel('Error rate')
    plt.show()

    return best_error

    # print(yhat1)


def CT():
    best_lambda, best_error = cross_validation(2)

    best_lambda = np.array(best_lambda).squeeze()
    out['l2'] = best_lambda
    best_error = np.array(best_error).squeeze()
    out['CT'] = best_error

    return best_error


def baseline():
    # Exercise 8_1_1
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))

    k = 0
    best_error = []
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        y_test_estimate = []

        # pred = np.bincount(np.reshape(X_test, X_test.size)).argmax()
        # print("Im here now2")
        # print(pred)
        '''error = np.sum(pred != y_test) / len(y_test)

        Error_test.append(error)'''

        largestCount = stats.mode(y_train)[0][0]
        # print("Im here now")
        # print(largestCount)

        y_test_estimate = np.ones((len(X_test))) * largestCount
        # print(y_test_estimate)

        yhat3.extend(y_test_estimate)
        Error_test[k] = np.sum(largestCount != y_test) / len(y_test)

        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        print('Test error: ', np.round(Error_test[k], 4))
        # print('Train indices: {0}'.format(train_index.size))
        # print('Test indices: {0}'.format(test_index.size))

        best_error.append(np.round(Error_test[k], 4))

        y_true.extend(y_test)

        k += 1

    best_error = np.array(best_error).squeeze()
    out['Baseline'] = best_error

    return Error_test


print("")
print("...........................................................")
print('Logistic regression starts here')
best_error1 = logistic_regression()

print("")
print("...........................................................")
print('CT starts here')
best_error2 = CT()

print("")
print("...........................................................")
print('Baseline model starts here')
best_error3 = baseline()

print(out)

y_true = np.array(y_true)
yhat1 = np.array(yhat1)
#print(yhat1.shape)
yhat2 = np.array(yhat2)
yhat3 = np.array(yhat3)


def model_vs_model(model1, model2):
    print("")
    print("Let the models loose")
    if model1 == 1:
        name1 = "Logistic regression"
        yhatModel1 = yhat1
    elif model1 == 2:
        name1 = "Classification trees"
        yhatModel1 = yhat2
    elif model1 == 3:
        name1 = "Baseline"
        yhatModel1 = yhat3

    if model2 == 1:
        name2 = "Logistic regression"
        yhatModel2 = yhat1
    elif model2 == 2:
        name2 = "Classification trees"
        yhatModel2 = yhat2
    elif model2 == 3:
        name2 = "Baseline"
        yhatModel2 = yhat3

    print(name1 + " vs. " + name2)
    [thetahat, CI, p] = mcnemar(y_true, yhatModel1, yhatModel2, alpha=0.05)
    print("theta: ", thetahat, " CI: ", CI, "p-value: ", p)


model_vs_model(1, 2)
model_vs_model(1, 3)
model_vs_model(2, 3)

legend = ["Log reg: ",
          "CT: ",
          "Baseline: "]



# x1 = range(len(baseline_y_tests[index]))
plt.plot([0,1,2,3,4,5,6,7,8,9], best_error1, 'b.-', [0,1,2,3,4,5,6,7,8,9], best_error2, 'r.-', [0,1,2,3,4,5,6,7,8,9], best_error3, 'g.-')
# plt.plot(x, lr_rest["pred"][index])
#plt.loglog(lambda_interval, test_error_rate2)
#plt.plot(lambda_interval, train_error_rate3)
plt.title("Comparison between predicted result and correct solution")
plt.xlabel('K-value')
plt.ylabel('Error rate')
#plt.xlim(0, len(lambda_interval))
plt.legend(legend)
plt.grid()
plt.show()
