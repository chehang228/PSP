
from fml import read_data
from fml.parameter_opt import GridSearch
from fml.data import DataObject
from fml.sampling import random_split
from sklearn.svm import SVR
import numpy as np, pandas as pd
from fml.validates import Validate
from sklearn.preprocessing import StandardScaler

split_result = read_data("logs.csv", df=True)
random_state = 231196
individuals = list(set([ int(i) for i in split_result.loc[str(random_state), "individuals"].split("-")]))
individuals = (np.array(individuals) + 1).tolist()
data = read_data("preprocessed_data.xlsx", df=False)
train, test = random_split(data, percent=0.2, random_state=random_state)
train = train.to_df().iloc[:, [0]+individuals]
test = test.to_df().iloc[:, [0]+individuals]
scaler = StandardScaler().fit(train.iloc[:, 1:])
train.iloc[:, 1:] = scaler.transform(train.iloc[:, 1:])
test.iloc[:, 1:] = scaler.transform(test.iloc[:, 1:])
train = DataObject().from_df(train)
test = DataObject().from_df(test)

gs = GridSearch().fit(SVR, train, test, cv=True, 
    **dict(
        C=np.linspace(10, 100, 91, endpoint=True),
        gamma=np.linspace(0.01, 1.01, 21, endpoint=True),
        epsilon=np.linspace(0.01, 1.01, 21, endpoint=True),
        ))

summuary = gs.results
summuary.to_excel("grid_results2.xlsx")
best_p = gs.best_p
print(best_p)

best_p = dict(
        C=24,
        gamma=0.06, 
        epsilon=0.06,
    )

v = Validate(SVR, train, test, **best_p).validate_all()
result = v.results

from joblib import dump
dump(dict(r=result, scaler=scaler, fnames=train.to_df().columns[1:]), "model")

data = read_data("preprocessed_data.xlsx", df=False)
train, test = random_split(data, percent=0.2, random_state=random_state)
train = train.to_df().iloc[:, [0]+individuals]
test = test.to_df().iloc[:, [0]+individuals]
train.to_excel("subtrain.xlsx")
test.to_excel("subtest.xlsx")
