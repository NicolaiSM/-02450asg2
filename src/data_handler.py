import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection





def get_data(K, onehot_classes, drop_columns, target):

    filename = '../data/flag_data'
    df = pd.read_csv(filename, 
                    sep=",",
                    names=["name", "landmass", "zone", "area", "population", "language", "religion", "bars", "stripes", "colours", "red", "green", "blue", "gold", "white", "black", "orange", "mainhue", "circles", "crosses", "saltires", "quarters", "sunstars", "crescent", "triangle", "icon", "animate", "text", "topleft", "botright"],
                    )


    for c in onehot_classes:
        onehotdf = pd.get_dummies(df[c], prefix=c)
        df = df.merge(onehotdf, left_index=True, right_index=True)

    data = df.drop(columns=drop_columns)

    data = data.values
    target = df[target].values


    N, M = data.shape

    #Laver det om til 0 mean og 1 std
    data = preprocessing.scale(data)
    target = preprocessing.scale(target)

    attributeNames = df.columns

    classNames = sorted(set(target))

    CV = model_selection.KFold(n_splits=K,shuffle=True)

    data_train = []
    target_train = []
    data_test = []
    target_test =  []

    data_train_outer = []
    target_train_outer = []
    data_test_outer = []
    target_test_outer =  []


    for train_index, test_index in CV.split(data, target):

        n_data_train = data[train_index]
        n_target_train = target[train_index]
        n_data_test = data[test_index]
        n_target_test = target[test_index]

        
        data_train.append(n_data_train)
        target_train.append(n_target_train)
        data_test.append(n_data_test)
        target_test.append(n_target_test)
        

        n_data_train_outer_list = []
        n_target_train_outer_list = []
        n_data_test_outer_list = []
        n_target_test_outer_list =  []
        


        for train_index_outer, test_index_outer in CV.split(n_data_train, n_target_train):

            n_data_train_outer_list.append(data[train_index_outer])
            n_target_train_outer_list.append(target[train_index_outer])
            n_data_test_outer_list.append(data[test_index_outer])
            n_target_test_outer_list.append(target[test_index_outer])
        

        data_train_outer.append(n_data_train_outer_list)
        target_train_outer.append(n_target_train_outer_list)
        data_test_outer.append(n_data_test_outer_list)
        target_test_outer.append(n_target_test_outer_list)


    return data, target, N, M, attributeNames, classNames, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer


"""
K=10

target = "colours"

drop_columns = ["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"]

onehot_classes = ["landmass", "zone", "language", "religion"]

data, target, N, M, attributeNames, CV, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer = get_data(K, onehot_classes, drop_columns, target)


print(np.array(data_train, dtype=object).shape)
print(np.array(target_train, dtype=object).shape)
print(np.array(data_test, dtype=object).shape)
print(np.array(target_test, dtype=object).shape)
print(np.array(data_train_outer, dtype=object).shape)
print(np.array(target_train_outer, dtype=object).shape)
print(np.array(data_test_outer, dtype=object).shape)
print(np.array(target_test_outer, dtype=object).shape)

print(np.array(data_train, dtype=object))
print(10*"-")
print(np.array(target_train, dtype=object))
print(10*"-")
print(np.array(data_test, dtype=object))
print(10*"-")
print(np.array(target_test, dtype=object))
print(10*"-")
print(np.array(data_train_outer, dtype=object))
print(10*"-")
print(np.array(target_train_outer, dtype=object))
print(10*"-")
print(np.array(data_test_outer, dtype=object))
print(10*"-")
print(np.array(target_test_outer, dtype=object))
print(10*"-")
"""


