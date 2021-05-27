import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import re
import unicodedata
import os

DATASETS = ['ECG200', 'GunPoint', 'ECGFiveDays', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Coffee', 'FaceFour', 'Ham', 'Herring', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MoteStrain', \
            'Plane', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Wine', 'ChlorineConcentration', 'Symbols', 'Wafer']


st.set_page_config(layout='wide')

def _get_dataset_config():
    # When executed with HTcondor dataset.json should be located at top level
    with open('datasets.json') as f:
            data = json.load(f)
    return data


def get_files(there: str, file_extension: str):
    """
    Returns csv files in directory 'there'.

    :param there: relative path
    :param file_extension:
    :return: list with csv files
    """
    path = there
    files = [path + x for x in os.listdir(path) if file_extension in x]
    return sorted(files)


def load_dataset(name: str):
    """
    Load dataset with config from CONFIG_FILE. Returns files alphabetically.
    :param name: dataset name
    :return: list of data frames
    """
    data_file = _get_dataset_config()
    dataset_dict = data_file[name]
    file_extension = dataset_dict['file_extension']
    files = get_files(dataset_dict['path'], file_extension)
    if not files:
        raise FileNotFoundError
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, **dataset_dict['csv_options'], dtype='float'))
    return dfs


def load_dataframe(name: str):
    """
    Load dataset with config from CONFIG_FILE.
    Returns dataframe with labels -1 for outlier and 1 for normals.
    :param name: dataset name
    :return: DataFrame
    """
    dfs = load_dataset(name)
    data_file = _get_dataset_config()
    dataset_dict = data_file[name]
    normal_labels = dataset_dict['normal_labels']
    if len(normal_labels) == 0:
        raise ValueError('dataset.json is invalid. Must contain at least one normal class label.')
    outlier_labels = dataset_dict['outlier_labels']
    if len(outlier_labels) == 0:
        raise ValueError('dataset.json is invalid. Must contain at least one outlier class label.')
    if 1000 in normal_labels or -1000 in outlier_labels:
        raise ValueError("You can't be serious!")
    for df in dfs:
        for normal_label in normal_labels:
            normal_index = df[df.iloc[:, 0] == float(normal_label)].index
            df.iloc[normal_index, 0] = 1000
        for outlier_label in outlier_labels:
            outlier_index = df[df.iloc[:, 0] == float(outlier_label)].index
            df.iloc[outlier_index, 0] = -1000
        normal_index = df[df.iloc[:, 0] == float(1000)].index
        df.iloc[normal_index, 0] = 1
        outlier_index = df[df.iloc[:, 0] == float(-1000)].index
        df.iloc[outlier_index, 0] = -1
    return pd.concat(dfs, ignore_index=True)


"""# Data set"""


"""**Choose data sets:**"""
data_set_selection = st.multiselect("", options=DATASETS, default=DATASETS)

normal_data_ratio = st.sidebar.slider("Percentage of normal instances in train data?", min_value=0.1, max_value=1.,
                                      value=.8, step=0.05)
outlier_ratio = st.sidebar.slider("Percentage outlier ratio in train data?", min_value=0.0, max_value=0.5,
                                  value=.05, step=0.01)

stats_df = pd.DataFrame()
for data_set in data_set_selection:
    df = load_dataframe(data_set)
    normal_index = df[df.iloc[:, 0] == 1].index
    outlier_index = df[df.iloc[:, 0] == -1].index
    n_normal_train = int(len(normal_index) * normal_data_ratio)
    n_normal_test = len(normal_index) - n_normal_train
    n_train = int(n_normal_train / (1 - outlier_ratio))
    n_outlier_train = n_train - n_normal_train
    n_outlier_test = len(outlier_index) - n_outlier_train
    stats = dict()
    stats['data_set'] = data_set
    stats['n_train'] = n_train
    stats['n_normal_train'] = n_normal_train
    stats['n_outlier_train'] = n_outlier_train
    stats['n_normal_test'] = n_normal_test
    stats['n_outlier_test'] = n_outlier_test
    stats['outlier_ratio_test'] = n_outlier_test / (n_outlier_test + n_normal_test)
    stats_df = stats_df.append(stats, ignore_index=True)

stats_df = stats_df.loc[:, ['data_set', 'n_train', 'n_normal_train', 'n_outlier_train',
                            'n_normal_test', 'n_outlier_test', 'outlier_ratio_test']]
st.dataframe(stats_df, width=1600, height=1000)

data_set_str = [f"'{data_set}'" for data_set in data_set_selection]
data_set_str = '[' + ', '.join(data_set_str) + ']'
st.write(data_set_str)






