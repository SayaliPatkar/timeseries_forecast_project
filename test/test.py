"""
This module handles deployment and testing of pretrained models

"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import explained_variance_score

from util import constants as const
from util import custom_logger
from util import result_filing
from util import load_data
from util import model_manipulation

file_id = const.TEST

def test_model(config):
    """The method loads model given model, test data and
    saves the test results at specified output directory.
    """
    run_id = run_id = config.info.run_id
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    test_df = load_data.load_from_tsv(config.train_test_split.test, 'testing')
    if not test_df.shape[0] == 0:
        logger.info('The data is loaded successfully')
    else:
        logger.error('Empty dataframe loaded')
        sys.exit('Empty dataframe loaded')
    print('Test df : %s'%(str(test_df.shape)))
    # m past smaples to consider for prediction
    m = config.info.m
    # n next steps to predict
    n = config.info.n
    model_file = config.info.model_file
    logger.info('Test is to predict next %d steps using past %d steps using model %s.'%(n,m,model_file))
    test_dict = dict()
    X_test, y_test = load_data.create_custom_data_structure(test_df, m, n)
    test_dict['X_test_shape']= X_test.shape
    test_dict['y_test_shape']= y_test.shape
    print('X_test : %s and y_test: %s '%(str(X_test.shape), str(y_test.shape)))
    has_null = y_test.isnull().sum().sum()  + X_test.isnull().sum().sum()
    if not has_null:
        logger.info('Successfuly built custom data structure for (%d input steps, %d output steps) supervised prediction'%(m,n))
    else:
        logger.error('Built custom dataframes have ', has_null ,' NaN values')
        sys.exit('Built custom dataframes have ', has_null ,' NaN values')
    model = model_manipulation.unpickle_model(model_file)
    y_test_predict = model.predict(X_test)
    avg_test_error = explained_variance_score(y_test, y_test_predict, multioutput='uniform_average')
    test_dict['avg_test_error']= avg_test_error
    result_filing.save_meta_file(test_dict, 'test_results')
    logger.info('Loaded model predicts with %d average validation error.'%(avg_test_error))
