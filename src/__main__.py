from data_handler import get_data
#from ann_reg import ANN_trainer
from baseline import baseline


drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language", "religion"]

data, target, N, M, attributeNames, CV = get_data(10, onehot_classes, drop_columns)

baseline(data, target, N, M, attributeNames, CV)

"""
n_hiddens = [1, 2, 4]
ann = ANN_trainer()
ann.train(data, target, N, M, attributeNames, CV, n_hiddens)
"""













