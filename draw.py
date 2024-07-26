# -*- coding:utf-8 -*-
# @author: Young
# @date: 2024-07-25 18:04:06

import os
import json
import matplotlib.pyplot as plt

def read_data(file_path:str):
    return json.load(open(file_path, 'r', encoding='utf-8'))

def draw(data:dict, upper_path: str):
    fig, ax = plt.subplots()

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

if __name__=="__main__":
    output_dir = "image/duizhao_xihu"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    data = read_data("results/duizhao_xihu/results.json")
    draw(data, output_dir)
