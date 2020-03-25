"""
This module is the staring point of the application.

"""
import yamlargparse
import sys
import pandas as pd

from util import constants as const
from util import custom_logger, result_filing
from train import train
from test import test

file_id = const.MAIN

def main():
    """
    The start of the flow handles all initializations, configuration loading
    and performs training or deployment based upon 'config.info.operation_type'
    For more details on configuration options look up commnets in config.yaml
    """
    parser = get_parser()
    config = parser.parse_args(['--cfg', 'config.yaml'])
    result_filing.init_config_vars(config)
    run_id = config.info.run_id
    logger = custom_logger.CustomLogger(run_id+':'+file_id)

    operation = config.info.operation_type
    logger.info("Selected operation type %s."%(operation))
    if operation == const.TRAIN_OP:
        train.train_model(config)
    elif operation == const.DEPLOY_OP:
        test.test_model(config)



def get_parser():
    """
    Loading parser to parse yaml configurations.
    """
    parser = yamlargparse.ArgumentParser(
        prog='train_forcast',
        description='configurations realted to training process of forcasting mechanism'
    )
    parser.add_argument('--info.run_id', default='',
                        help='the unique identifier for logging and metadata creation')
    parser.add_argument('--info.m', default=10,
                        help='use past m values for prediction')
    parser.add_argument('--info.n', default=5,
                        help='predict next n values')
    parser.add_argument('--info.operation_type',
                        choices=[const.TRAIN_OP, const.DEPLOY_OP],
                        help='choosing whether to perform training or deployment')
    parser.add_argument('--info.model_type',
                        choices=[const.LIN_REG, const.RAN_FOR_REG, const.DEC_TREE_REG, const.MULT_OP_REG],
                        help='choosing model type in case of training operation')
    parser.add_argument('--info.model_file', default='',
                        help='the relative path to the stored model file')
    parser.add_argument('--info.output_dir', default='output',
                        help='the relative path to the directory for storing results')
    parser.add_argument('--train_test_split.type',
                        choices=[const.SPLIT_BY_DATE, const.SPLIT_BY_FILES],
                        help='determines the way in which train-test split should be done')
    parser.add_argument('--train_test_split.date', default='',
                        help='the date string in \'YYYY-mm-dd\' format, indicating the date at which split should be made')
    parser.add_argument('--train_test_split.train', default='',
                        help='the relative path to the .tsv file containing train data')
    parser.add_argument('--train_test_split.test', default='',
                        help='the relative path to the .tsv file containing test data')
    parser.add_argument('--visualize.train_data', action=yamlargparse.ActionYesNo, default=False,
                        help='determines if the training visualizations are to be stored')
    parser.add_argument('--visualize.train_fname', default='',
                        help='the relative path to the .pdf file storing train data visualizations')
    parser.add_argument('--random_forest_regression.max_depth', default=20,
                        help='choosing hyperparams for random forest')
    parser.add_argument('--random_forest_regression.random_state', default=7,
                        help='choosing hyperparams for random forest')
    parser.add_argument('--decison_tree_regression.max_depth', default=20,
                        help='choosing hyperparams for decision tree')
    parser.add_argument('--multi_output_regression.n_estimators', default=100,
                        help='choosing hyperparams for multioutput regression')

    parser.add_argument('--cfg', action=yamlargparse.ActionConfigFile, required=True)
    return parser


if __name__ == '__main__':
    main()
