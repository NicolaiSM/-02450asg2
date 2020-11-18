from data_handler import get_data
from ann_reg import ANN_trainer
from baseline import baseline

target = "colours"

drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language", "religion"]

data, target, N, M, attributeNames, CV = get_data(10, onehot_classes, drop_columns, target)

base_line_error = baseline(data, target, N, M, attributeNames, CV)

print(base_line_error)

n_hiddens = [1, 5, 10, 20, 40, 80]
ann = ANN_trainer()

#ann.run(data, target, M, CV, n_hiddens)
print(ann.innerrun(CV, data, target, M, n_hiddens))


print(ann.n_hidden)
print(ann.error)












