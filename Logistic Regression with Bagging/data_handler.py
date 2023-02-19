import pandas as pd
import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    data = pd.read_csv('data_banknote_authentication.csv')
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1:].to_numpy().flatten()
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    if(shuffle):
        shuffler = np.random.permutation(len(X))
        X = X[shuffler]
        y = y[shuffler]

    X_train = X[:int(len(X)*(1-test_size))]
    y_train = y[:int(len(X)*(1-test_size))]

    X_test = X[int(len(X)*(1-test_size)):]
    y_test = y[int(len(X)*(1-test_size)):]
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    idx_list = list(range(X.shape[0]))
    rand_idx = np.random.choice(idx_list, X.shape[0])
    X_sample, y_sample = X[rand_idx], y[rand_idx]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample


