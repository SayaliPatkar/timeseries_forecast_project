"""
This module handles training-validation split related tasks

"""
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta

from util import constants as const
from util import custom_logger
from util import load_data


file_id = const.SPLIT_TRAIN

def split_by_date(run_id, train_fname, split_date):
    """
    The method splits the training data by date
    """
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    if split_date == '':
        logger.error('No split date provided')
        sys.exit('No split date provided')
    df_intraday = load_data.load_from_tsv(train_fname, 'training')
    date_start = str(df_intraday.head(1).index.date[0])
    date_end = str(df_intraday.tail(1).index.date[0])
    train_df = df_intraday[date_start:split_date]
    date_split_1 = datetime.strptime(split_date, "%Y-%m-%d")+ timedelta(days=1)
    test_df =  df_intraday[date_split_1:date_end]
    return train_df, test_df

def split_by_file(run_id, train_fname, test_fname):
    """
    The method splits the training data by file
    """
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    train_df = load_data.load_from_tsv(train_fname, 'training')
    test_df = load_data.load_from_tsv(test_fname, 'test')
    return train_df, test_df
