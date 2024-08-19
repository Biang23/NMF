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

from draw import draw_W, draw_W_all

root_dir = "data"
child_dirs = os.listdir(root_dir)
result_dir = "results"

w_3 = {}
h_3 = {}

def vaf(V: np.ndarray, Vr: np.ndarray) -> float:
    return 1 - (np.sum((V-Vr)**2)/np.sum(V**2))

def regularization(matrix: np.ndarray):
    
    epsilon = 1e-5
    min_vals = matrix.min(axis=0)  
    max_vals = matrix.max(axis=0)  
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals + epsilon)

    return normalized_matrix

def nmf(matrix, group, person):
    matrix = np.transpose(matrix)
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

        WR = regularization(W)
        HR = regularization(H)
        ACTIVATION = np.mean(regularization(WR @ HR), axis=0)

        wh = {
            i: dict(
                W=str(W),
                WR=str(WR),
                H=str(H),
                HR=str(HR),
                VR=str(VR),
                VAF=VAF,
                ACTIVATION=str(ACTIVATION)
            )
        }
        if i == 3 or i == 4:
            draw_W(WR, group, person)

        if i == 3:
            if group not in w_3.keys():
                w_3[group] = W
            else:
                w_3[group] += W
            
            if group not in h_3.keys():
                h_3[group] = H.transpose()[:160, :]
            else:
                h_3[group] += H.transpose()[:160, :]
        
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
            result[person.split(".")[0]].append(nmf(matrix, group, person))

        # Draw W&H all fig
        for group in w_3.keys():
            w_all = [w_3[group][:, i]/16 for i in range(w_3[group].shape[1])]
            h_all = [h_3[group][:, i]/16 for i in range(h_3[group].shape[1])]
            draw_W_all(w_all, h_all, group)

        output_dir = os.path.join(result_dir, group)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    main()
    
