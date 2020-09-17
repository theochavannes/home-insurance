import os
import sys
import logging

LOGGING_FORMAT = "%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class Config:
    DATA_FOLDER = os.getenv("HOME_INSURANCE_DATA_FOLDER", "C:\\data\\esure\\home-insurance")
    DATA_NAME = os.getenv("HOME_INSURANCE_DATA_NAME", "home_insurance.csv")
    DATA_PATH = os.path.join(DATA_FOLDER, DATA_NAME)
    AUTOML_FOLDER = os.path.join(DATA_FOLDER, "automl")
    DATA_ANALYSIS_FOLDER = os.path.join(DATA_FOLDER, "data_analysis")
    OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")

def set_logging():
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)


config = Config()
set_logging()
