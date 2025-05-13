# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict
import h5py
import pandas as pd
import shutil

# +
SRC_USER = "mengdixu"
DST_USER = "chengshu"
ROOT_DIR = "/vision/u"
PROJECT = "mimicgen"

TASK = "pick_cup"
DR = "D0"
ABLATIONS = ["full"]
# ABLATIONS = ["no_vis", "only_hard", "only_soft", "full"]

for ABLATION in ABLATIONS:
    FOLDER = f"{TASK}_{ABLATION}"
    if TASK == "tidy_table" and ABLATION == "full":
        PATH = f"{ROOT_DIR}/{SRC_USER}/{PROJECT}/{FOLDER}_old"
    else:
        PATH = f"{ROOT_DIR}/{SRC_USER}/{PROJECT}/{FOLDER}"
    for folder in sorted(os.listdir(PATH)):
        print(ABLATION, folder)
        folder_path = os.path.join(PATH, folder)
        for subfolder in sorted(os.listdir(folder_path)):
            if DR not in subfolder: continue
            stats_json_file = os.path.join(folder_path, subfolder, "subtask_lengths.json")
            demo_hdf5_file = os.path.join(folder_path, subfolder, "demo.hdf5")
            if os.path.isfile(stats_json_file): continue
            subtask_lengths = dict()
            if not os.path.isfile(demo_hdf5_file): continue
            print(demo_hdf5_file)
            with h5py.File(demo_hdf5_file, "r") as hdf5_f:
                for demo_key in hdf5_f["data"]:
                    subtask_lengths[demo_key] = hdf5_f["data"][demo_key]["subtask_lengths"][:].tolist()
            print("num of demos stats collected", len(subtask_lengths))
            with open(stats_json_file, "w+") as f:
                json.dump(subtask_lengths, f)
