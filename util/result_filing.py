"""
This module handles saving and visualization tasks for matadata and plots

"""
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from util import constants as const
from util import custom_logger

file_id = const.VISUALIZE

def init_config_vars(config):
    """
        Initializes global variables from yaml config, once every main() call
    """
    global run_id
    run_id = config.info.run_id
    global unique_op_dir
    unique_op_dir = os.path.join(config.info.output_dir, config.info.run_id)
    os.makedirs(unique_op_dir, exist_ok=True)


def visualize_train_data(train_df, fname):
    """
    Visualize the time series input
    """
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    fig, axs = plt.subplots(3, figsize=(15,15))
    fig.suptitle('EPEX Intraday Continuous market electricity prices')

    axs[0].plot(train_df.index, train_df['low'],  color='red')
    axs[0].set_title("Lowest Price")
    axs[0].set(xlabel='time', ylabel='price (Euros)')

    axs[1].plot(train_df.index, train_df['high'],  color='green')
    axs[1].set_title("Highest Pice")
    axs[1].set(xlabel='time', ylabel='price (Euros)')

    axs[2].plot(train_df.index, train_df['weight_avg'],  color='blue')
    axs[2].set_title("volume-weighted Average Price")
    axs[2].set(xlabel='time', ylabel='price (Euros)')

    fig.savefig(os.path.join(unique_op_dir, fname))
    logger.info('Training data plots stored at ', os.path.join(unique_op_dir, fname))

def save_meta_file(gen_dict, f_name):
    """
    Write train_model or test_results metadata to specified file
    """
    logger = custom_logger.CustomLogger(run_id+':'+file_id)
    filename = run_id+'_'+ f_name +'.meta'
    f = open(os.path.join(unique_op_dir, filename),'a')
    print('Output stored in %s'%(str(os.path.join(unique_op_dir, filename))))
    logger.info('Output stored in %s'%(str(os.path.join(unique_op_dir, filename))))
    for key, val in gen_dict.items():
        line = str(key)+" : "+str(val)+"\n"
        f.write(line)
