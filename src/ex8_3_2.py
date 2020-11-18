# exercise 8.3.2 Fit multinomial regression
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import sklearn.linear_model as lm
from data_handler import get_data

# Load Matlab data file and extract variables of interest
'''mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]

N, M = X.shape
C = len(classNames)'''
#%% Model fitting and prediction


K = 10

drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red",
                "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language", "religion"]

X, yDontCare, y, N, M, attributeNames, CV = get_data(K, onehot_classes, drop_columns)

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
for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]


    # Multinomial logistic regression
    logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-08, random_state=1)
    logreg.fit(X_train,y_train)

    # To display coefficients use print(logreg.coef_). For a 4 class problem with a
    # feature space, these weights will have shape (4, 2).

    print(sum(logreg.predict(X_test)!=y_test))
    # Number of miss-classifications
    print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))

    predict = lambda x: np.argmax(logreg.predict_proba(x),1)
    figure(2,figsize=(9,9))
    #visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, attributeNames)
    title('LogReg decision boundaries')

    show()

    print('Ran Exercise 8.3.2')