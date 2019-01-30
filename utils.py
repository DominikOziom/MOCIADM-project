# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def prepare_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    features = data.iloc[:, 0:-1].values.astype(float)
    labels = data.iloc[:, -1].values.astype(str)
    classes = np.unique(labels)
    return features, labels, classes


def load_prepare_data(filename):
    data = pd.read_csv(filename)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    features = data.iloc[:, 0:-1].values.astype(float)
    labels = data.iloc[:, -1].values.astype(str)
    classes = np.unique(labels)
    return features, labels, classes
