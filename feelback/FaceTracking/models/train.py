#!/usr/bin/env python3

import numpy as np
import sys
import pickle
import os
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

name = ""
dataset = None

for dataset_name in sys.argv:
    if not dataset_name.endswith(".npy"):
        continue
    
    if dataset is None:
        dataset = np.load(dataset_name)
    else:
        dataset = np.append(dataset, np.load(dataset_name), axis=0)
    
    dataset_name = os.path.basename(dataset_name)
    dataset_name = re.sub(r'_\d+x\d+', '', dataset_name)
    name += os.path.splitext(dataset_name)[0] + "+"

name = name.rstrip('+')

for n in [50]:
    model = make_pipeline(MinMaxScaler(), PCA(n))
    model.fit(dataset)

    pickle.dump(model, open(f"{name}_n={n}_minmax.model", "wb"))
