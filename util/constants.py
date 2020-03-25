"""This file stores constants
"""
# training or deployment
TRAIN_OP = 'training'
DEPLOY_OP = 'deployment'

# type of model to be built for TRAIN_OP
LIN_REG = 'linear_regression'
RAN_FOR_REG = 'random_forest_regression'
DEC_TREE_REG = 'decison_tree_regression'
MULT_OP_REG = 'multi_output_regression'

#for test train split in TRAIN_OP
SPLIT_BY_DATE = 'by_date'
SPLIT_BY_FILES = 'by_files'
STRPTIME_FORMAT = '%Y-%m-%d'

# for adding file name to logging
MAIN = 'main.py'
TRAIN = 'train\\train.py'
TEST = 'test\\test.py'
SPLIT_TRAIN = 'train\\train_test_split.py'
VISUALIZE = 'utils\\result_filing.py'
LOAD = 'utils\\load_data.py'
LOAD_M = 'utils\\load_model.py'
