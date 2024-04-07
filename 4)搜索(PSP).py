
import pandas as pd
from fml.descriptors import Atom
import joblib
from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
from pathlib import Path
import numpy as np

model_results = joblib.load("model")
model = model_results["r"]["train"]["model"]
scaler = model_results["scaler"]
fnames = model_results["fnames"]
atoms = pd.read_excel("atoms.xlsx").columns.values.tolist()
criterion = 0.0001
iteration = 10000
atom_gen = Atom(other=True)
target = 8

logpath = f"search_logs_{target}.csv"
if not Path(logpath).exists():
    with open(logpath, "w") as f:
        f.writelines(f"formular, error, absolute_error, prediction, {','.join(fnames.tolist())}\n")
    existed_formulars = []
else:
    existed_formulars = pd.read_csv(logpath).iloc[:, 0].values.tolist()

# 范围自己调, 元素范围见atoms.xlsx
hspace = [
    hp.choice("A_ele", ["Au"]),
    hp.choice("B_ele", atoms),
    hp.choice("C_ele", atoms),
    hp.choice("A_ratio", [1]),
    hp.uniform("B_ratio", 0, 5),
    hp.uniform("C_ratio", 0, 5),
    ]

def searching_fn(params):
    
    formular = []
    sample_descriptors = []
    ratios = np.array(params[3:])
    atoms = np.array(params[:3])
    ratios = ratios / ratios.sum()
    ratios = ratios.round(3)
    ratios2 = ratios / ratios[0]
    ratios2 = ratios2.round(3)
    
    ratio_descriptors = pd.Series(ratios, index=["Au", "B_ratio", "C_ratio"])
    for atom, ratio, ratio2 in zip(atoms, ratios, ratios2):
        atom_descriptors = atom_gen.describe(atom, onehot=True) * float(ratio)
        atom_descriptors.name = atom
        sample_descriptors.append(atom_descriptors)
        formular.append(f"{atom}{ratio2}")
    sample_descriptors = pd.concat(sample_descriptors, axis=1)
    sample_descriptors_sum = sample_descriptors.sum(axis=1)
    sample_descriptors_sum = pd.concat([ratio_descriptors, sample_descriptors_sum])
    sample_descriptors_sum = sample_descriptors_sum[fnames]
    formular = "".join(formular)
    
    sample_descriptors_sum_scaled = pd.DataFrame(sample_descriptors_sum).T * 1
    
    sample_descriptors_sum_scaled.iloc[:, :] = scaler.transform(
        sample_descriptors_sum_scaled
        )
    
    prediction = model.predict(sample_descriptors_sum_scaled.values)[0]
    error = abs(prediction - target) ** 0.5
    if formular not in existed_formulars:
        with open(logpath, "a") as f:
            f.writelines(f"{formular}, {prediction - target}, {abs(prediction - target)}, {prediction}, {', '.join(sample_descriptors_sum.values.astype(str).tolist())}\n")
    return error, formular

def f(params):
    error = searching_fn(params)[0]
    return {'loss': error, 'status': STATUS_OK}

trials = Trials()

best = fmin(
    fn=f, 
    space=hspace,
    algo=tpe.suggest, 
    max_evals=iteration,
    trials=trials,
    )

