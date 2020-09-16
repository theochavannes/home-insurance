import numpy as np
np.random.seed(123)
from home_insurance.config import config
import pandas as pd
from aikit.ml_machine import MlMachineLauncher
from home_insurance.constants import DATES_COLS
from sklearn.model_selection import ShuffleSplit
from aikit.cross_validation import IndexTrainTestCv
from collections import OrderedDict
import os

def loader():
    """ modify this function to load the data
 à)
    Returns
    -------
    dfX, y

    Or
    dfX, y, groups

    """
    data = pd.read_csv(os.path.join(config.DATA_FOLDER, "data_for_automl.csv"))
    dfX, y = data.drop(["POL_STATUS"], axis=1), (data["POL_STATUS"]=="Lapsed").astype(int)
    return dfX, y

def set_configs(launcher):
    """ this is the function that will set the different configurations """
    # Change the CV here :
    launcher.job_config.cv = IndexTrainTestCv(np.arange(int(190000*0.75)))

    # Change the scorer to use :
    launcher.job_config.scoring = ['accuracy', 'log_loss_patched', 'avg_roc_auc', 'f1_macro']

    # Change the main scorer (linked with )
    launcher.job_config.main_scorer = 'avg_roc_auc'

    # Change the base line (for the main scorer)
    launcher.job_config.score_base_line = 0.84

    # Allow 'approx cv or not :
    launcher.job_config.allow_approx_cv = True

    # Allow 'block search' or not :
    launcher.job_config.do_blocks_search = True

    # Start with default models or not :
    launcher.job_config.start_with_default = True

    # Specify the type of problem
    launcher.auto_ml_config.type_of_problem = 'CLASSIFICATION'


if __name__ == "__main__":
    launcher = MlMachineLauncher(base_folder=config.AUTOML_FOLDER,
                                 name="esure_home_insurance",
                                 loader=loader,
                                 set_configs=set_configs)

    launcher.execute_processed_command_argument()