# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-07-24 17:01:47

import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np

from tqdm import tqdm

from sklearn.decomposition import NMF

root_dir = "data"
child_dirs = os.listdir(root_dir)
result_dir = "results"

def vaf(V: np.ndarray, Vr: np.ndarray) -> float:
    return 1 - (np.sum((V-Vr)**2)/np.sum(V**2))

def nmf(matrix):

    res = []

    matrix = np.abs(matrix)
    
    if (matrix<0).any(): 
        return res

    for i in range(1, 11):
        model = NMF(n_components=i, init='random', random_state=0, max_iter=10000)

        W = model.fit_transform(matrix)
        H = model.components_
        VR = W @ H
        VAF = vaf(matrix, VR)

        wh = {
            i: dict(
                W=str(W),
                H=str(H),
                VR=str(VR),
                VAF=VAF
            )
        }

        res.append(wh)
    return res

def main():
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for group in child_dirs:
        result = defaultdict(list)
        persons = os.listdir(os.path.join(root_dir, group))
        for person in tqdm(persons, total=len(persons), desc=f"{group} processing"):
            cur_file_name = os.path.join(root_dir, group, person)
            df = pd.read_excel(cur_file_name)
            df = df.iloc[:, df.shape[1]-10:df.shape[1]]
            matrix = df.values
            # print(cur_file_name, matrix)
            # raise ValueError("test")
            result[person.split(".")[0]].append(nmf(matrix))

        output_dir = os.path.join(result_dir, group)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    main()

