"""
This module handles training and validation task

"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score

from util import constants as const
from util import custom_logger, load_data, model_manipulation
from train import train_test_split

file_id = const.TRAIN

def train_model(config):
    """The method loads the training data, performs test_train split.
    Trains the specific type of model with specific hyperparameters (as selected in config)
    and saves the model and metadata at specified output directory.
    For additional configuration info, look up documentation of config.yaml
    """
    run_id = config.info.run_id
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # More split and/or corssvailation approaches can be added to
    # from train.train_test_split.py
    split_type = config.train_test_split.type
    if split_type == const.SPLIT_BY_DATE:
        split_date = config.train_test_split.date
        train_file_name = config.train_test_split.train
        train_df, test_df = train_test_split.split_by_date(run_id, train_file_name, split_date)
    elif split_type == const.SPLIT_BY_FILES:
        train_file_name = config.train_test_split.train
        test_file_name = config.train_test_split.test
        train_df, test_df = train_test_split.split_by_file(run_id, train_file_name, test_file_name)

    if not (train_df.shape[0] == 0 and test_df.shape[0] == 0):
        logger.info('The data is loaded successfully')
    else:
        logger.error('Empty dataframe loaded')
        sys.exit('Empty dataframe loaded')

    # m past smaples to consider for prediction
    m = config.info.m
    # n next steps to predict
    n = config.info.n
    # which model to train
    model = config.info.model_type
    # this dictionary will be stored as model metadata
    model_info = dict()
    model_info['model_type']= model
    model_info['past_m']= m
    model_info['next_n']= n
    logger.info('Selected %s model to predict next %d steps using past %d steps.'%(model,n,m))
    print('Train df : %s'%(str(train_df.shape)))
    print('Test df : %s'%(str(test_df.shape)))
    # converting loaded dataframe to consider past m values and predict next n
    # all the rows having NaN values are dropped
    X_train, y_train = load_data.create_custom_data_structure(train_df, m, n)
    X_test, y_test = load_data.create_custom_data_structure(test_df, m, n)
    model_info['X_train_shape']= X_train.shape
    model_info['y_train_shape']= y_train.shape
    model_info['X_test_shape']= X_test.shape
    model_info['y_test_shape']= y_test.shape
    print('X_train : %s and y_train: %s '%(str(X_train.shape), str(y_train.shape)))
    print('X_test : %s and y_test: %s '%(str(X_test.shape), str(y_test.shape)))
    has_null = X_train.isnull().sum().sum() + y_train.isnull().sum().sum() + y_test.isnull().sum().sum()  + X_test.isnull().sum().sum()
    if not has_null:
        logger.info('Successfuly built custom data structure for (%d input steps, %d output steps) supervised prediction'%(m,n))
    else:
        logger.error('Built custom dataframes have ', has_null ,' NaN values')
        sys.exit('Built custom dataframes have ', has_null ,' NaN values')

    if model == const.LIN_REG:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_test_predict = lr_model.predict(X_test)
        # Since this is multi_output regression problem, more sophisticated
        # metrics can be calculated, also there is overlap between predicted values
        # as next n predictions are done for sliding past m values, new metrics
        # should justify this sliding window structure too
        avg_error = explained_variance_score(y_test, y_test_predict, multioutput='uniform_average')
        model_info['validation_avg_error']= avg_error
        model_manipulation.pickle_model(lr_model, model_info)
        logger.info('Selected %s model predicts with %d average validation error.'%(model,avg_error))

    elif model == const.RAN_FOR_REG:
        model_info['max_depth']= config.random_forest_regression.max_depth
        model_info['random_state']= config.random_forest_regression.random_state

        rf_model= RandomForestRegressor(max_depth=model_info['max_depth'],
                                        random_state=model_info['random_state'])
        rf_model.fit(X_train, y_train)
        y_test_predict = rf_model.predict(X_test)
        # Since this is multi_output regression problem, more sophisticated
        # metrics can be calculated, also there is overlap between predicted values
        # as next n predictions are done for sliding past m values, new metrics
        # should justify this sliding window structure too
        avg_error = explained_variance_score(y_test, y_test_predict, multioutput='uniform_average')
        model_info['validation_avg_error']= avg_error
        model_manipulation.pickle_model(rf_model, model_info)
        logger.info('Selected %s model predicts with %d average validation error.'%(model,avg_error))

    elif model == const.DEC_TREE_REG:
        model_info['max_depth']= config.decison_tree_regression.max_depth
        dt_model= DecisionTreeRegressor(max_depth=model_info['max_depth'])
        dt_model.fit(X_train, y_train)
        y_test_predict = dt_model.predict(X_test)
        # Since this is multi_output regression problem, more sophisticated
        # metrics can be calculated, also there is overlap between predicted values
        # as next n predictions are done for sliding past m values, new metrics
        # should justify this sliding window structure too
        avg_error = explained_variance_score(y_test, y_test_predict, multioutput='uniform_average')
        model_info['validation_avg_error']= avg_error
        model_manipulation.pickle_model(dt_model, model_info)
        logger.info('Selected %s model predicts with %d average validation error.'%(model,avg_error))                    # fit() with instantiated object

    elif model == const.MULT_OP_REG:
        model_info['n_estimators']= config.multi_output_regression.n_estimators
        mor_model= MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=model_info['n_estimators']))
        mor_model.fit(X_train, y_train)
        y_test_predict = mor_model.predict(X_test)
        # Since this is multi_output regression problem, more sophisticated
        # metrics can be calculated, also there is overlap between predicted values
        # as next n predictions are done for sliding past m values, new metrics
        # should justify this sliding window structure too
        avg_error = explained_variance_score(y_test, y_test_predict, multioutput='uniform_average')
        model_info['validation_avg_error']= avg_error
        model_manipulation.pickle_model(mor_model, model_info)
        logger.info('Selected %s model predicts with %d average validation error.'%(model,avg_error))
