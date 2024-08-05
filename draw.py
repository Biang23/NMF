# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-07-25 18:04:06

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path:str):
    return json.load(open(file_path, 'r', encoding='utf-8'))

def draw_vaf(data:dict, upper_path: str):
    fig, ax = plt.subplots(figsize=(12,12))
    print(type(ax))
    for name, d in data.items():
        x, y = [], []
        for item in d[0]:
            key, value = int(list(item.keys())[0]), list(item.values())[0]['VAF']
            x.append(key)
            y.append(value)
        
        if not len(x) == len(y) == 10:
            print("Error")

        ax.plot(x, y, label=name)

    ax.set_title(f"VAF of {upper_path.split('/')[-1]}")
    ax.set_xlabel("n_components")
    ax.set_ylabel("VAF")   
    ax.legend()
    plt.savefig(os.path.join(upper_path, f"{upper_path.split('/')[-1]}.jpg"))

def draw_W(w: np.ndarray, group: str, person: str):
    w = np.transpose(w)
    y = np.arange(1, w.shape[1]+1)
    plot_num = w.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=plot_num, sharey=True)

    for i in range(w.shape[0]):
        color_map = ['grey' if x < 0.5 else 'black' for x in list(w[i][:])]
        ax[i].barh(y, list(w[i][:]), height=0.5, color=color_map, tick_label=y)
    fig.suptitle(f"{group.split('_')[0]}_{person.split('.')[0]}_W{plot_num}")
    plt.savefig(f"image/{group}/{person.split('.')[0]}_W_{plot_num}.jpg")
    plt.close()


if __name__=="__main__":
    output_dir = "image/duizhao_xihu"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    data = read_data("results/duizhao_xihu/results.json")
    draw_vaf(data, output_dir)
