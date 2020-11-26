# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for project_fraud Project
"""

from os.path import split
import pandas as pd
import datetime
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)

# function to clean the data


def merge_data(path_to_data):
    # argument of function --> path to the folder where you have stored the cvs's

    #import data
    train_identity = pd.read_csv(path_to_data + 'train_identity.csv')
    train_transaction = pd.read_csv(path_to_data + 'train_transaction.csv')

    #merge
    df_merged = train_transaction.merge(train_identity, on='TransactionID', how='left')

    return df_merged


def drop_many_missing_values():
    df = merge_data('~/data/')

    # create list of numerical and categorical features
    c = (df.dtypes == 'object')
    n = (df.dtypes != 'object')
    cat_cols = list(c[c].index)
    num_cols = list(n[n].index)

    # for cat: create low, medium, many missing values lists
    many_missing_cat_cols = []     # more than 60% missing
    for i in cat_cols:
        percentage = df[i].isnull().sum() * 100 / len(df[i])
        if percentage > 60:
            many_missing_cat_cols.append(i)
    # drop categorical features with many missing values
    df = df.drop(columns = many_missing_cat_cols)

    # for num: create low, medium, many missing values lists
    many_missing_num_cols = []     # more than 60% missing
    for i in num_cols:
        percentage = df[i].isnull().sum() * 100 / len(df[i])
        if percentage > 60:
            many_missing_num_cols.append(i)
    # drop features with many missing values
    df = df.drop(columns = many_missing_num_cols)

    return df


#if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    #import project_fraud
    #folder_source, _ = split(project_fraud.__file__)
    #df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    #print(' dataframe cleaned')
