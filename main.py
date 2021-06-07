import streamlit as st
import pandas as pd
from src import SessionState
from src.utils import Context, load_config, load_dataset, get_data_set_info

# To add more data sets, extend this list and add the files to data/ as well ass an entry to datasets.json.
DATASETS = ['ECG200', 'GunPoint', 'ECGFiveDays', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Coffee', 'FaceFour', 'Ham', 'Herring', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MoteStrain', \
            'Plane', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Wine', 'ChlorineConcentration', 'Symbols', 'Wafer']


st.set_page_config(layout='wide')

# Required to save state between widget interactions.
session_state = SessionState.get(configuration=Context())

"""# Time series classification data set to outlier data set assistant"""
"""This tool assists one in transforming a binary or multiclass classification data set such that it is usable in an anomaly detection task."""
"""A straightforward approach is to select a class as _normal_ and regard all other classes as outliers ([Emmott 2013](https://dl.acm.org/doi/abs/10.1145/2500853.2500858)).
For a binary data set, we choose the majority class as the normal class. For a multiclass data set, we choose the visually most distinct class.
This creates diverse outliers."""

"""The first section of the tool assists in choosing the normal and outlying classes. The second section summarizes the amount of normal and outlying time series."""

## Sidebar section
st.sidebar.markdown('1. Transformation')
data_set_selection = st.sidebar.selectbox("Choose data set:", options=DATASETS)

display_data_sets_placeholder = st.sidebar.empty()
configured_data_sets = display_data_sets_placeholder.multiselect("Configured data sets:", default=session_state.configuration.get_data_sets(), options=session_state.configuration.get_data_sets())

st.sidebar.markdown('___')
st.sidebar.markdown('2. Summary')
normal_data_ratio = st.sidebar.slider("Percentage of normal instances in train data.", min_value=0.1, max_value=1.,
                                      value=.8, step=0.05)

outlier_ratio = st.sidebar.slider("Percentage outlier ratio in train data.", min_value=0.0, max_value=0.5,
                                  value=.05, step=0.01)
st.sidebar.markdown('___')
if st.sidebar.button("Load example configuration", help="Loads a sample configuration for all data sets."):
    for data_set, normal_label, outlier_labels in load_config():
        data_set_config = dict()
        data_set_config['normal_class'] = normal_label[0]
        data_set_config['outlier_classes'] = outlier_labels
        data_set_config = pd.Series(data_set_config, name=data_set)
        session_state.configuration.append(data_set_config)
    configured_data_sets = display_data_sets_placeholder.multiselect("Configured data sets:", default=session_state.configuration.get_data_sets(), options=session_state.configuration.get_data_sets())

## Main section 1.
"## 1. Transformation"

session_state.configuration.update(configured_data_sets)

time_series, class_numbers, stats = get_data_set_info(load_dataset(data_set_selection))

col1, col2 = st.beta_columns(2)
with col1:
    "Choose a data set from the side panel."
    st.line_chart(time_series.T)
with col2:
    "Some stats for the displayed time series classes."
    st.table(stats)
    "Choose the normal class:"
    normal_class_selection = st.selectbox("", class_numbers)
    if st.button('Add as normal class', help="Select the chosen class as the normal class and add it to the configuration."):
        data_set_config = dict()
        data_set_config['normal_class'] = normal_class_selection
        class_numbers.remove(normal_class_selection)
        data_set_config['outlier_classes'] = class_numbers
        data_set_config = pd.Series(data_set_config, name=data_set_selection)
        session_state.configuration.append(data_set_config)
        configured_data_sets = display_data_sets_placeholder.multiselect("Configured data sets:", default=session_state.configuration.get_data_sets(), options=session_state.configuration.get_data_sets())

## Main section 2.
"## 2. Summary"
"Use the sliders in the side panel to regulate the normal and outlier ratios."
stats_df = pd.DataFrame()
if len(configured_data_sets) > 0:
    for data_set in configured_data_sets:
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


