"""
This module handles loading and storage of trained models

"""
import os
import sys
import pickle
from util import constants as const
from util import custom_logger, result_filing

file_id = const.LOAD_M

def unpickle_model(model_fanme):
    """
    Load model from 'config.info.model_file'
    """
    print(model_fanme)
    logger = custom_logger.CustomLogger(result_filing.run_id+':'+file_id)
    if model_fanme == '':
        logger.error('Filepath for model is not provided')
        sys.exit('Filepath for model is not provided')
    else:
        loaded_model = pickle.load(open(model_fanme, 'rb'))
        return loaded_model

def pickle_model(model, model_info):
    """
    Store model and model metadata in specified file'
    """
    run_id = result_filing.run_id
    unique_op_dir = result_filing.unique_op_dir
    filename_1 = run_id+'_'+model_info['model_type']+'.pkl'
    pickle.dump(model, open(os.path.join(unique_op_dir, filename_1), 'wb'))
    result_filing.save_meta_file(model_info, model_info['model_type'])
