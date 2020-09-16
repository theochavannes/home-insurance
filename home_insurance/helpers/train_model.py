import pandas as pd
from home_insurance.config import config
from home_insurance.core.model import Predictor
from aikit.tools.helper_functions import save_pkl
import os

if __name__=="__main__":
    data = pd.read_csv(config.DATA_PATH)
    predictor = Predictor()
    predictor.train(data)
    save_pkl(predictor, os.path.join(predictor.model_folder, predictor.get_name(ext="pickle")))
