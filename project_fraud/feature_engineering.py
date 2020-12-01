import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression


def clean_mail(data):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other',
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other',
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    for c in ['P_emaildomain']:
        data[c + '_bin'] = data[c].map(emails)
        data[c + '_suffix'] = data[c].map(lambda x: str(x).split('.')[-1])

    return data[c + '_bin'], data[c + '_suffix']


def make_day_feature(data, offset=0, column_name='TransactionDT'):

    days = data[column_name] / (3600*24)
    encoded_days = np.floor(days-1+offset) % 7
    return encoded_days


def make_hour_feature(data, column_name='TransactionDT'):

    hours = data[column_name] / (3600)
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


def string_card(row):
    if pd.isna(row['card1']):
        return np.nan
    elif pd.isna(row['card2']):
        return np.nan
    elif pd.isna(row['card3']):
        return np.nan
    elif pd.isna(row['card4']):
        return np.nan
    elif pd.isna(row['card5']):
        return np.nan
    elif pd.isna(row['card6']):
        return np.nan
    else:
        return str(row['card1']) + str(row['card2']) + str(row['card3']) + str(row['card4']) + str(row['card5']) + str(row['card6'])


def credit_cards(data):
    credit_cards = data.groupby('cardID').agg(
        mean = pd.NamedAgg(column='TransactionAmt', aggfunc='mean'),
        min = pd.NamedAgg(column='TransactionAmt', aggfunc='min'),
        max = pd.NamedAgg(column='TransactionAmt', aggfunc='max'),
        median = pd.NamedAgg(column='TransactionAmt', aggfunc='median'),
    )
    return credit_cards


def dist_from_mean(row, metric):
    if pd.isna(row['TransactionAmt']):
        return np.nan
    if pd.isna(row[metric]):
        return np.nan
    else:
        dist = row['TransactionAmt'] - row[metric]
        return dist


def dist_from_median_rel(row, metric):
    if pd.isna(row['TransactionAmt']):
        return np.nan
    if pd.isna(row[metric]):
        return np.nan
    else:
        dist_rel = (row['TransactionAmt'] - row[metric]) / row[metric]
        return dist_rel

# def clean_mail_suffix(P_emaildomain_suffix):
#    if P_emaildomain_suffix == 'nan':
#       return np.nan
#    else:
#        return P_emaildomain_suffix


### final function

def transform_raw_data(data):
    data = drop_many_missing_values()
    data['P_emaildomain_bin'], data['P_emaildomain_suffix'] = clean_mail(data)
    #data_transformed['P_emaildomain_suffix'] = data_transformed['P_emaildomain_suffix'].apply(lambda x: clean_mail_suffix(x))
    data['weekday'] = make_day_feature(data, offset=0.58)
    data['hours'] = make_hour_feature(data)
    data['cardID'] = data.apply(lambda row: string_card(row), axis=1)
    data = data.merge(credit_cards(data), how='left', on="cardID")
    data['dist_mean'] = data.apply(lambda row: dist_from_mean(row, 'mean'), axis=1)
    data['dist_median'] = data.apply(lambda row: dist_from_mean(row, 'median'), axis=1)
    data['dist_mean_rel'] = data.apply(lambda row: dist_from_median_rel(row, 'mean'), axis=1)
    data['dist_median_rel'] = data.apply(lambda row: dist_from_median_rel(row, 'median'), axis=1)
    return data








