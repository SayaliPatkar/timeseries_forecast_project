"""
This module handles logging

"""
import logging
import os
from util import result_filing

class LoggerAdapter(logging.LoggerAdapter):
    """
    To create logs with unique run_id
    """
    def __init__(self, logger, prefix):
        super(LoggerAdapter, self).__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return '[%s] %s' % (self.prefix, msg), kwargs

class CustomLogger:
    """
    Customized logging solution
    """
    def __init__(self, app_name):
        logger = logging.getLogger(__name__)
        #unique_op_dir is global variable
        cust_handler = logging.FileHandler(os.path.join(result_filing.unique_op_dir, "logs.log"))
        cust_formatter = logging.Formatter('%(asctime)s::%(levelname)s:: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        cust_handler.setFormatter(cust_formatter)
        logger.addHandler(cust_handler)
        self.logger = LoggerAdapter(logger, app_name)

    def info(self, msg):
        """
        INFO log
        """
        self.logger.setLevel(logging.INFO)
        self.logger.info(msg)

    def debug(self, msg):
        """
        DEBUG log
        """
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(msg)

    def warn(self, msg):
        """
        WARNING log
        """
        self.logger.setLevel(logging.WARNING)
        self.logger.warning(msg)

    def error(self, msg):
        """
        ERROR log
        """
        self.logger.setLevel(logging.ERROR)
        self.logger.error(msg)

    def critical(self, msg):
        """
        CRITICAL log
        """
        self.logger.setLevel(logging.CRITICAL)
        self.logger.critical(msg)
