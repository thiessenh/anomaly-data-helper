
import json
import os
import pandas as pd
def _get_dataset_config():
    """Loads JSON that contains settings how to read the data sets.

    Returns
    -------
    dict
        Access settings with data set name as key.
    """
    with open('datasets.json') as f:
            data = json.load(f)
    return data


def get_files(there: str, file_extension: str):
    """Returns csv files in directory 'there'.

    Parameters
    ----------
    there : str
            Relative path.
    file_extension : str
            File extension such as 'csv' or 'tsv'.
    Returns
    -------
    list
        List of files that match file extension.
    """
    path = there
    files = [path + x for x in os.listdir(path) if file_extension in x]
    return sorted(files)


def load_dataset(name: str):
    """Load dataset with config from CONFIG_FILE. Returns files alphabetically.

    Parameters
    ----------
    name : str
            Name of data set.

    Returns
    -------
    pd.DataFrame
        Concatenation of data frames.

    Raises
    ------
    FileNotFoundError
        Raised when directory is empty.
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
    """Load example data set config.

    Yields
    -------
    tuple
        Tuple containing the data set name, normal class value, and outlier class values.

    Raises
    ------
    ValueError
        Raised when normal class value is not exactly one, or when no outlier labels provided.
    """
    with open('datasets.json') as f:
        data_file = json.load(f)
        for name in data_file.keys():
            dataset_dict = data_file[name]
            normal_label = dataset_dict['normal_class']
            if len(normal_label) == 0:
                raise ValueError('dataset.json is invalid. Must contain at least one normal class label.')
            outlier_labels = dataset_dict['outlier_classes']
            if len(outlier_labels) == 0:
                raise ValueError('dataset.json is invalid. Must contain at least one outlier class label.')
            yield name, normal_label, outlier_labels


def get_data_set_info(data:pd.DataFrame):
    """Returns one time series per class and some stats along with it.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame containg the time series. The first column is expected to contain the class label.

    Returns
    -------
    tuple
        Data frame with time series and a data frame with stats.
    """
    time_series = []
    std = []
    n_time_series = []
    min = []
    max = []
    for class_number in data.iloc[:, 0].unique():
        class_data = data[data.iloc[:, 0] == class_number]
        tmp = class_data.iloc[0]
        time_series.append(pd.Series(tmp, name=f"Class: {class_number}"))
        # get some stats
        std.append(class_data.std(axis=0).median())
        n_time_series.append(class_data.shape[0])
        min.append(class_data.min(axis=0).median())
        max.append(class_data.max(axis=0).median())
    stats = pd.DataFrame([std,n_time_series,min,max])
    stats = stats.T
    stats.columns = ['standard deviation', '# time series', 'min', 'max']
    stats.index = list(map(lambda x: f"Class: {x}", data.iloc[:, 0].unique()))
    return pd.DataFrame(time_series), data.iloc[:, 0].unique().tolist(), stats

class Context:
    """Class holding a data frame that contains the data set transformation information, i.e, the normal and outlier labels.
    """
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