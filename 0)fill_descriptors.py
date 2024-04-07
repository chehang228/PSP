

import pandas as pd, numpy as np
import re
from fml.descriptors import Atom

data = pd.read_excel("data.xlsx")
formulars = data.iloc[:, 0].tolist()
atom = Atom(other=True)

fd = []
splited_formulars = []
with open("./split_results.csv", "w") as f:
    for for_index, formular in enumerate(formulars):
        write_str = ""
        sf = re.compile("[A-Z]{1}[a-z]{1,2}[\d+\.]+|[A-Z]{1}[\d+\.]+|[A-Z]{1}[a-z]{1,2}|[A-Z]{1}").findall(formular)
        
        splited_formulars.append(sf)
        
        sample_descriptors = []
        ratio_sum = 0
        ratios = []
        for sf_ in sf:
            ele = re.compile("[A-Z]{1}[a-z]{1,2}|[A-Z]{1}").findall(sf_)[0]
            ratio = re.compile("[\d+\.]+").findall(sf_)
            if len(ratio) == 0:
                ratio = 1
            else:
                ratio = ratio[0]
            # write_str += f"{ele}, {ratio}, "
            ratio_sum += float(ratio)
            
            atom_descriptors = atom.describe(ele, onehot=True) * float(ratio)
            atom_descriptors.name = ele
            sample_descriptors.append(atom_descriptors)
            ratios.append(ratio)
        
        sample_descriptors = pd.concat(sample_descriptors, axis=1) / ratio_sum
        # sample_descriptors.to_excel(f"./tmp/{for_index+1}_{formular}.xlsx")
        sample_descriptors_sum = sample_descriptors.sum(axis=1)
        ratios = pd.Series(ratios, index=["A_ratio", "B_ratio", "C_ratio"])
        sample_descriptors_sum = pd.concat([ratios, sample_descriptors_sum])
        fd.append(sample_descriptors_sum)
        
        # f.writelines(f"{write_str}\n")

fd = pd.concat(fd, axis=1).T
fd = pd.concat([data, fd], axis=1)
fd.index = fd.iloc[:, 0]
fd = fd.iloc[:, 1:]
fd = fd.astype(float)
fd.to_excel("filled_descriptors.xlsx")
