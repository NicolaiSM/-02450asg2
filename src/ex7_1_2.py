import scipy.stats as ss
from ex7_1_1 import *


def jeffrey_interval(y, yhat, alpha):
    m = sum(y - yhat == 0)
    n = y.size
    a = m+.5
    b = n-m + .5
    CI = ss.beta.interval(1-alpha, a=a, b=b)
    thetahat = a/(a+b)
    return thetahat, CI

# Compute the Jeffreys interval
alpha = 0.05
for i in range(len(L)):
    [thetahatA, CIA] = jeffrey_interval(y_true, yhat[:, i], alpha=alpha)
    print("Theta point estimate", thetahatA, " CI: ", CIA)
