import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

filename = '/Users/nicolaimikkelsen/Desktop/02450asg2/data/flag_data'
df = pd.read_csv(filename, 
                sep=",",
                names=["name", "landmass", "zone", "area", "population", "language", "religion", "bars", "stripes", "colours", "red", "green", "blue", "gold", "white", "black", "orange", "mainhue", "circles", "crosses", "saltires", "quarters", "sunstars", "crescent", "triangle", "icon", "animate", "text", "topleft", "botright"],
                )


onehot_classes = ["landmass", "zone", "language", "religion"]


for c in onehot_classes:
    onehotdf = pd.get_dummies(df[c], prefix=c)
    df = df.merge(onehotdf, left_index=True, right_index=True)


data = df.drop(columns=["name", "mainhue", "topleft", "botright", "landmass", "zone", "language", "religion", "colours", "red", "green", "blue", "gold", "white", "black", "orange"])

data = data.values
target = df["colours"].values


N, M = data.shape

#Laver det om til 0 mean og 1 std
data = preprocessing.scale(data)
target = preprocessing.scale(target)

attributeNames = df.columns



def get_data(K):
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    return data, target, N, M, attributeNames, CV