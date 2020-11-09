# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:43:27 2020

@author: Feyza
"""

import pickle
import pandas as pd

path_pickles = "C:/Users/Feyza/Desktop/denemetez/latest-dneme/04. Model Training/Model/"

list_pickles = [
    "df_models_gbc.pickle",
    "df_models_knnc.pickle",
    "df_models_mnbc.pickle",
    "df_models_rfc.pickle",
    "df_models_svc.pickle"
]

df_summary = pd.DataFrame()

for pickle_ in list_pickles:
    
    path = path_pickles + pickle_
    
    with open(path, 'rb') as data:
        df = pickle.load(data)

    df_summary = df_summary.append(df)

df_summary = df_summary.reset_index().drop('index', axis=1)


df_summary

df_summary.sort_values('Test Set Accuracy', ascending=False)
