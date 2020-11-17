from data_handler import get_data
#from ann_reg import ANN_trainer
from baseline import baseline

data, target, N, M, attributeNames, CV = get_data(10)

print(baseline(data, target, N, M, attributeNames, CV))

"""
n_hiddens = [1, 2, 4]
ann = ANN_trainer()
ann.train(data, target, N, M, attributeNames, CV, n_hiddens)
"""













