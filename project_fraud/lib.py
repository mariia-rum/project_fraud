# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for project_fraud Project
"""

from os.path import split
import pandas as pd
import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

pd.set_option('display.width', 200)

# function to clean the data


def import_concat_data(path_to_data):
    # argument of function --> path to the folder where you have stored the cvs's

    #import data
    train_identity = pd.read_csv(path_to_data + 'train_identity.csv')
    train_transaction = pd.read_csv(path_to_data + 'train_transaction.csv')
    test_identity = pd.read_csv(path_to_data + 'test_identity.csv')
    test_transaction = pd.read_csv(path_to_data + 'test_transaction.csv')

    # delete target in train_transaction
    train_transaction = train_transaction.drop('isFraud', axis=1)

    # rename columns in test_identity
    test_identity.columns = ['TransactionID','id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
       'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
       'DeviceInfo']

    # concate
    transaction_data = pd.concat([train_transaction, test_transaction])
    identity_data = pd.concat([train_identity, test_identity])

    return transaction_data, identity_data


def merge_data():
    transaction_data, identity_data = import_concat_data('~/data/')
    df_merged = transaction_data.merge(identity_data, on='TransactionID', how='left')
    return df_merged


def clean_merge_data():
    # use import_concat_data function
    transaction_data, identity_data = import_concat_data('~/data/')

    # create list of numerical and categorical features

    ## for transaction_data ##
    c = (transaction_data.dtypes == 'object')
    n = (transaction_data.dtypes != 'object')
    cat_trans_cols = list(c[c].index)
    num_trans_cols = list(n[n].index)
    # create low, medium, many missing values lists
    # for cat
    low_missing_cat_trans_cols = []      # lower than 15% missing values
    medium_missing_cat_trans_cols = []   # between 15% and 60% missing
    many_missing_cat_trans_cols = []     # more than 60% missing
    for i in cat_trans_cols:
        percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
        if percentage < 15:
            low_missing_cat_trans_cols.append(i)
        elif percentage >= 15 and percentage < 60:
            medium_missing_cat_trans_cols.append(i)
        else:
            many_missing_cat_trans_cols.append(i)
    # for num
    low_missing_num_trans_cols = []      # lower than 15% missing values
    medium_missing_num_trans_cols = []   # between 15% and 60% missing
    many_missing_num_trans_cols = []     # more than 60% missing
    for i in num_trans_cols:
        percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
        if percentage < 15:
            low_missing_num_trans_cols.append(i)
        elif percentage >= 15 and percentage < 60:
            medium_missing_num_trans_cols.append(i)
        else:
            many_missing_num_trans_cols.append(i)

    ## for identity_data ##
    c = (identity_data.dtypes == 'object')
    n = (identity_data.dtypes != 'object')
    cat_id_cols = list(c[c].index)
    num_id_cols = list(n[n].index)
    # create low, medium, many missing values lists
    # for cat
    low_missing_cat_id_cols = []      # lower than 15% missing values
    medium_missing_cat_id_cols = []   # between 15% and 60% missing
    many_missing_cat_id_cols = []     # more than 60% missing
    for i in cat_id_cols:
        percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
        if percentage < 15:
            low_missing_cat_id_cols.append(i)
        elif percentage >= 15 and percentage < 60:
            medium_missing_cat_id_cols.append(i)
        else:
            many_missing_cat_id_cols.append(i)
    # for num
    low_missing_num_id_cols = []      # lower than 15% missing values
    medium_missing_num_id_cols = []   # between 15% and 60% missing
    many_missing_num_id_cols = []     # more than 60% missing
    for i in num_id_cols:
        percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
        if percentage < 15:
            low_missing_num_id_cols.append(i)
        elif percentage >= 15 and percentage < 60:
            medium_missing_num_id_cols.append(i)
        else:
            many_missing_num_id_cols.append(i)

    # clean NUMERICAL features

    # drop numerical features with many missing values
    identity_data = identity_data.drop(columns = many_missing_num_id_cols)

    transaction_data = transaction_data.drop(columns = many_missing_num_trans_cols)


    # impute numerical features with medium missing values
    my_imputer = SimpleImputer(strategy = 'median')
    my_imputer.fit(identity_data[medium_missing_num_id_cols])
    identity_data[medium_missing_num_id_cols] = my_imputer.transform(identity_data[medium_missing_num_id_cols])

    my_imputer = SimpleImputer(strategy = 'median')
    my_imputer.fit(transaction_data[medium_missing_num_trans_cols])
    transaction_data[medium_missing_num_trans_cols] = my_imputer.transform(transaction_data[medium_missing_num_trans_cols])

    # impute numerical features with low missing values
    my_imputer = SimpleImputer(strategy = 'mean')
    my_imputer.fit(identity_data[low_missing_num_id_cols])
    identity_data[low_missing_num_id_cols] = my_imputer.transform(identity_data[low_missing_num_id_cols])

    my_imputer = SimpleImputer(strategy = 'mean')
    my_imputer.fit(transaction_data[low_missing_num_trans_cols])
    transaction_data[low_missing_num_trans_cols] = my_imputer.transform(transaction_data[low_missing_num_trans_cols])


    # clean CATEGORIAL features

    # drop features with many missing values
    identity_data = identity_data.drop(columns = many_missing_cat_id_cols)

    transaction_data = transaction_data.drop(columns = many_missing_cat_trans_cols)

    # merge
    df_merged = transaction_data.merge(identity_data, on='TransactionID', how='left')

    return df_merged




if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import project_fraud
    folder_source, _ = split(project_fraud.__file__)
    df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    clean_data = clean_data(df)
    print(' dataframe cleaned')
