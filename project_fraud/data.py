# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for project_fraud Project
"""

from os.path import split
import pandas as pd
import datetime
import numpy as np


from project_fraud.feature_engineering import clean_mail
from project_fraud.feature_engineering import make_day_feature
from project_fraud.feature_engineering import make_hour_feature
from project_fraud.feature_engineering import string_card
from project_fraud.feature_engineering import credit_cards
from project_fraud.feature_engineering import dist_from_mean
from project_fraud.feature_engineering import dist_from_median_rel


pd.set_option('display.width', 200)

# function to clean the data


def cleaned_featured_data(path_to_data):

    #import data
    train_identity = pd.read_csv(path_to_data + 'train_identity.csv')
    train_transaction = pd.read_csv(path_to_data + 'train_transaction.csv')

    #merge
    df = train_transaction.merge(train_identity, on='TransactionID', how='left')

    # get rid of nan values

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

    # features
    df['P_emaildomain_bin'], df['P_emaildomain_suffix'] = clean_mail(df)
    df['weekday'] = make_day_feature(df, offset=0.58)
    df['hours'] = make_hour_feature(df)
    df['cardID'] = df.apply(lambda row: string_card(row), axis=1)
    df = df.merge(credit_cards(df), how='left', on="cardID")
    df['dist_mean'] = df.apply(lambda row: dist_from_mean(row, 'mean'), axis=1)
    df['dist_median'] = df.apply(lambda row: dist_from_mean(row, 'median'), axis=1)
    df['dist_mean_rel'] = df.apply(lambda row: dist_from_median_rel(row, 'mean'), axis=1)
    df['dist_median_rel'] = df.apply(lambda row: dist_from_median_rel(row, 'median'), axis=1)

    return df


#if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    #import project_fraud
    #folder_source, _ = split(project_fraud.__file__)
    #df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    #print(' dataframe cleaned')
