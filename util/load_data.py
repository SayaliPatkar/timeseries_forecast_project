"""
This module handles loading and manipulation of dataframes

"""
import sys
import os
import pandas as pd
from util import custom_logger, result_filing
from util import constants as const

file_id = const.LOAD

def load_from_tsv(filename, type):
    """
    The method loads tsv file
    """
    #run_id is global variable
    logger = custom_logger.CustomLogger(result_filing.run_id+':'+file_id)
    if filename == '':
        logger.error('Filepath for %s is not provided'%(type))
        sys.exit('Filepath for %s is not provided'%(type))
    else:
        loaded_df = pd.read_csv(filename, sep="\t", infer_datetime_format=True, parse_dates=['delivery_start'], index_col=['delivery_start'])
        # nan check
        has_null = loaded_df.isnull().sum().sum()
        if not has_null:
            logger.info('No NaN values in dataframe loaded from %s'%(filename))
        else:
            logger.error('Dataframe loaded from %s has %d NaN values'%(filename, has_null))
            sys.exit('Dataframe loaded from %s has %d NaN values'%(filename, has_null))
    return loaded_df


def create_custom_data_structure(data_df, m, n):
    """
    The method converts loaded dataframe to consider past m values and predict next n values
    for supervised learning type format.
    Input m*3 features, output n*3 features, 3 is for 'low', 'high' and 'weighted_avg'
    All the rows having NaN values are dropped.
    """
    orig_cols = data_df.columns
    cols, names = list(), list()
    # input sequence (t-m, ... t-1) for all original variables
    for i in range(m, 0, -1):
        cols.append(data_df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in orig_cols]
    X_df = pd.concat(cols, axis=1)
    # discarding first m rows for which there are no earlier m X_df values
    # and last n rows for which there are no naxt n y_df values
    X_df = X_df[m:-n]
    X_df.columns = names

    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n) for all original variables
    for i in range(0, n):
        cols.append(data_df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in orig_cols]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in orig_cols]
    y_df = pd.concat(cols, axis=1)
    # discarding first m rows for which there are no earlier m X_df values
    # and last n rows for which there are no naxt n y_df valuesv
    y_df = y_df[m:-n]
    y_df.columns = names
    return X_df, y_df
