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
    return pd.concat(dfs)


def load_config():
    """
    Load example data set config.
    """
    data_file = _get_dataset_config()
    for name in data_file.keys():
        dataset_dict = data_file[name]
        normal_labels = dataset_dict['normal_labels']
        if len(normal_labels) == 0:
            raise ValueError('dataset.json is invalid. Must contain at least one normal class label.')
        outlier_labels = dataset_dict['outlier_labels']
        if len(outlier_labels) == 0:
            raise ValueError('dataset.json is invalid. Must contain at least one outlier class label.')
        if 1000 in normal_labels or -1000 in outlier_labels:
            raise ValueError("You can't be serious!")
        yield name, normal_labels, outlier_labels

def get_classes_to_plot(data:pd.DataFrame):
    time_series = []
    std = []
    n = []
    min = []
    max = []
    for class_number in data.iloc[:, 0].unique():
        class_data = data[data.iloc[:, 0] == class_number]
        tmp = class_data.iloc[0]
        time_series.append(pd.Series(tmp, name=f"Class: {class_number}"))
        # get some stats
        std.append(class_data.std(axis=0).median())
        n.append(class_data.shape[0])
        min.append(class_data.min(axis=0).median())
        max.append(class_data.max(axis=0).median())
    stats = pd.DataFrame([std,n,min,max])
    stats = stats.T
    stats.columns = ['standard deviation', '# time series', 'min', 'max']
    stats.index = list(map(lambda x: f"Class: {x}", data.iloc[:, 0].unique()))
    return pd.DataFrame(time_series), data.iloc[:, 0].unique().tolist(), stats

class Context:
    def __init__(self) -> None:
        self.configurations = pd.DataFrame()
        default = pd.Series({"normal_class": 1, "outlier_classes": [-1]}, name='ECG200')
        self.configurations = self.configurations.append(default)

    def append(self, data_set_config:pd.Series):
        try:
            self.configurations = self.configurations.append(data_set_config, verify_integrity=True)
        except ValueError:
            pass

    def update(self, display_selection:list):
        self.configurations = self.configurations.loc[display_selection, :]
    
    def get_data_sets(self) -> list:
        return self.configurations.index.tolist()

    def get_outlier_classes(self, data_set):
        return self.configurations.loc[data_set, 'outlier_classes']
    
    def get_normal_class(self, data_set):
        return self.configurations.loc[data_set, 'normal_class']

    def __repr__(self) -> str:
        return self.configurations.to_json(orient='index', indent=2)
    def clear(self):
        self.configurations = pd.DataFrame()





"""# Time series classification data set to outlier data set assistant"""
"""This tool assits one in transforming a binary or multiclass classifiaction data set such that it is usable in an anomlay detection task."""
st.markdown("""A straightforward appraoch is to select a class as _normal_ and regard all other classes as outliers ([Emmott 2013](https://dl.acm.org/doi/abs/10.1145/2500853.2500858)).
For a binary data set, we choose the majority class as normal class. For a multiclass data set, we choose the visually most distinct class.
This creates diverse outliers.""")

"""The first section of the tool assists in choosing the normal and outlying classes. The second section summarizes the amount of normal and outlying time series."""
import SessionState

session_state = SessionState.get(configuration=Context())

st.sidebar.markdown('1. Transformation')
data_set_selection = st.sidebar.selectbox("Choose data set:", options=DATASETS)

display_data_sets_placeholder = st.sidebar.empty()
display_data_set_selection = display_data_sets_placeholder.multiselect("Configured data sets:", default=session_state.configuration.get_data_sets(), options=session_state.configuration.get_data_sets())

st.sidebar.markdown('___')
st.sidebar.markdown('2. Summary')
normal_data_ratio = st.sidebar.slider("Percentage of normal instances in train data?", min_value=0.1, max_value=1.,
                                      value=.8, step=0.05)

outlier_ratio = st.sidebar.slider("Percentage outlier ratio in train data?", min_value=0.0, max_value=0.5,
                                  value=.05, step=0.01)
st.sidebar.markdown('___')
if st.sidebar.button("Load example configuration"):
    for data_set, normal, outliers in load_config():
        data_set_config = dict()
        data_set_config['normal_class'] = normal[0]
        data_set_config['outlier_classes'] = outliers
        data_set_config = pd.Series(data_set_config, name=data_set)
        session_state.configuration.append(data_set_config)


"## 1. Transformation"

session_state.configuration.update(display_data_set_selection)

time_series, class_numbers, stats = get_classes_to_plot(load_dataset(data_set_selection))

col1, col2 = st.beta_columns(2)
with col1:
    "Choose a data set from the sidepanel."
    st.line_chart(time_series.T)
with col2:
    "Some stats for the displayed time series classes."
    st.table(stats)
    "Choose the normal class:"
    normal_class_selection = st.selectbox("", class_numbers)
    if st.button('Add as normal class'):
        data_set_config = dict()
        data_set_config['normal_class'] = normal_class_selection
        class_numbers.remove(normal_class_selection)
        data_set_config['outlier_classes'] = class_numbers
        data_set_config = pd.Series(data_set_config, name=data_set_selection)
        session_state.configuration.append(data_set_config)
        display_data_set_selection = display_data_sets_placeholder.multiselect("Configured data sets:", default=session_state.configuration.get_data_sets(), options=session_state.configuration.get_data_sets())

st.dataframe(session_state.configuration.configurations)
"## 2. Summary"
"Use the sliders in the sidepanel to regulate the normal and outlier ratios."
stats_df = pd.DataFrame()
if len(display_data_set_selection) > 0:
    for data_set in display_data_set_selection:
        df = load_dataset(data_set)
        normal_index = df[df.iloc[:, 0] == session_state.configuration.get_normal_class(data_set)].index
        outlier_index = df[df.iloc[:, 0].isin(session_state.configuration.get_outlier_classes(data_set))].index
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
        stats['normal_class'] = session_state.configuration.get_normal_class(data_set)
        stats['outlier_classes'] = session_state.configuration.get_outlier_classes(data_set)
        stats_df = stats_df.append(stats, ignore_index=True)

    stats_df = stats_df.loc[:, ['data_set', 'n_train', 'n_normal_train', 'n_outlier_train',
                                'n_normal_test', 'n_outlier_test', 'outlier_ratio_test', 'normal_class', 'outlier_classes']]
    stats_df.columns = ['Data set', '# train', '# normal train', '# outlier train', '# normal test', '# outlier test', 'outlier ratio', 'normal class', 'outlier classes']
    st.dataframe(stats_df, width=1600, height=1000)

    with st.beta_expander("Show configuration as JSON"):
        st.code(repr(session_state.configuration), language='json')


