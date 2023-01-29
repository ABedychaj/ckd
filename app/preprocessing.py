import numpy as np
import pandas as pd

# From documentation of the dataset
NumericalColumns = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'pcv', 'hemo', 'wbcc', 'rbcc', ]
CategoricalColumns = ['al', 'su', 'rbc', 'sg', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


def clean_data(path):
    """
    Clean the data.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned data.
    """
    # Read the data
    df = pd.read_csv(path, index_col=0)

    df[CategoricalColumns] = df[CategoricalColumns].astype("object")
    df[NumericalColumns] = df[NumericalColumns].apply(pd.to_numeric)

    # We could use different techniques to fill NaNs:
    # * most frequent values for categorical and mean for numerical (most reasonable in our case)
    # * delete rows with missing values (if we do that we will lose a lot of data)
    # * we could create generative model to fill missing values (not enough data to train that model with)

    # Fill NaNs with most frequent values
    for columnName in CategoricalColumns:
        df[columnName].fillna(df[columnName].mode()[0], inplace=True)

    for columnName in NumericalColumns:
        df[columnName].fillna(df[columnName].mean(), inplace=True)
    return df


def encode_categorical(df):
    """
    Encode the categorical data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data.

    Returns
    -------
    df : pandas.DataFrame
        Encoded data.
    """
    # We could use different techniques to encode categorical variables:
    # * label encoding
    # * one-hot encoding
    # * target encoding
    # * etc.

    # We will use one-hot encoding
    df = pd.get_dummies(df, columns=CategoricalColumns, prefix=CategoricalColumns, drop_first=True)
    return df


def scale_numerical(df):
    """
    Scale the numerical data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data.

    Returns
    -------
    df : pandas.DataFrame
        Encoded data.
    """
    # We could use different techniques to scale numerical variables:
    # * min-max scaling
    # * standardization
    # * etc.

    # We will use min-max scaling
    for columnName in NumericalColumns:
        df[columnName] = (df[columnName] - df[columnName].min()) / (df[columnName].max() - df[columnName].min())
    return df


def encoding(df):
    """
    Encode the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data.

    Returns
    -------
    df : pandas.DataFrame
        Encoded data.
    """

    df = encode_categorical(df)
    df = scale_numerical(df)

    return df


def prepare_train_test_split(df):
    """
    Prepare train test split.

    Parameters
    ----------
    df : pandas.DataFrame
        Encoded data.

    Returns
    -------
    X_train : pandas.DataFrame
        Train data.
    X_test : pandas.DataFrame
        Test data.
    y_train : pandas.DataFrame
        Train targets.
    y_test : pandas.DataFrame
        Test targets.
    """
    X = df.loc[:, df.columns != 'class']
    y = df['class']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1990)

    return X_train, X_test, y_train, y_test
