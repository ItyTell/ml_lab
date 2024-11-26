import pandas as pd
import numpy as np

def get_data():
    data = pd.read_csv('data/data 2.csv')
    data['target'] = data['target'].map({'yes': 1, 'no': 0})
    data.drop(columns=['f16', 'f15', 'f14', 'f13', 'f12', 'f10', 'p'], inplace=True)
    return data

def clean_data(data):
    columns = list(data.columns)
    columns.remove('target')
    original_len = len(data)
    contradictions = data.groupby(columns)['target'].agg(['nunique', 'count'])
    contradictory_features = contradictions[contradictions['nunique'] > 1].index
    contradictory_mask = data.set_index(columns).index.isin(contradictory_features)
    num_contradictory = contradictory_mask.sum()
    data = data[~contradictory_mask]
    num_duplicates = data.duplicated().sum()
    data = data.drop_duplicates()
    print(f"Removed {num_contradictory / original_len * 100}% contradictory rows and {num_duplicates/original_len *100}% duplicates")
    return data
