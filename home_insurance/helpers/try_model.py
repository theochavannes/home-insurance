"""
This file is used to try the model and simulate having it delivered as an API. (takes a json as an input and it
outputs a dictionary)
"""
import json
import logging
import os

import pandas as pd
from aikit.tools.helper_functions import load_pkl

from home_insurance.config import config
from home_insurance.core.model import Predictor


def generate_try_data(name="try"):
    # choose the row number of the dataset you want to try
    input = pd.read_csv(config.DATA_PATH).iloc[10:20]
    input = input.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
    input = input.drop(["PAYMENT_FREQUENCY"], axis=1)
    input = input.drop(["POL_STATUS"], axis=1)
    logging.info("Trying input:\n {}".format(input))
    input.to_json(os.path.join(config.DATA_FOLDER, "{}.json".format(name)), orient="records")


if __name__ == "__main__":
    name = "try"
    generate_try_data(name)

    input = json.load(open(os.path.join(config.DATA_FOLDER, "{}.json".format(name)), "rb"))
    if not isinstance(input, list):
        input = [input]
    ##########################################################################
    # write the model id of the trained predictor you wanna try:
    ##########################################################################
    model_id = "20200916_171005"
    model: Predictor = load_pkl(os.path.join(config.OUTPUT_FOLDER,
                                             "home_insurance_{}".format(model_id),
                                             "model",
                                             "home_insurance_{}.pickle".format(model_id)))

    results = model.predict(input)
    logging.info("Output: {}".format(results))
