import os
import logging
import json_tricks

from aikit.tools.helper_functions import load_pkl
import json
from home_insurance.config import config
from home_insurance.core.model import Predictor

import pandas as pd

def generate_try_data(name="try"):
    input = pd.read_csv(config.DATA_PATH).iloc[0].to_frame().T
    input = input.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
    input = input.drop(["PAYMENT_FREQUENCY"], axis=1)
    input = input.drop(["POL_STATUS"], axis=1)
    logging.info("Trying input:\n {}".format(input))
    input.to_json(os.path.join(config.DATA_FOLDER, "{}.json".format(name)), orient="records")

if __name__=="__main__":
    name = "try"
    generate_try_data(name)

    input = json.load(open(os.path.join(config.DATA_FOLDER, "{}.json".format(name)), "rb"))
    if not isinstance(input, list):
        input = [input]

    df = pd.DataFrame.from_records(input)

    model_name = "demo_model.pkl"
    #write the folder with the correct model id you wanna try:
    model_id = "20200916_144220"
    model: Predictor = load_pkl(os.path.join(config.OUTPUT_FOLDER,
                                             "home_insurance_{}".format(model_id),
                                             "model",
                                             "home_insurance_{}.pickle".format(model_id)))
    results = model.predict(df)
    logging.info("Output: {}".format(results))