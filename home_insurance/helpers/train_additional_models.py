"""
This file cis used to build a baseline model (logistic regression) and one very slow model (svm), which
can not be run in the automl because of the time taken to fit it. It is still good to try it just in case?
"""
import numpy as np

np.random.seed(123)
from home_insurance.config import config
import pandas as pd
from sklearn.svm import SVC
import os
from sklearn.metrics import roc_auc_score
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from home_insurance.pipeline import pipeline

if __name__ == "__main__":
    dict_scores = {}
    preprocess = pipeline.get_subpipeline(end_node="PassThrough")
    path = os.path.join(config.DATA_FOLDER, "data_for_automl.csv")
    data = pd.read_csv(path)

    X, y = data.drop(["POL_STATUS"], axis=1), (data["POL_STATUS"] == "Lapsed").astype(int)
    X = preprocess.fit_transform(X, y)
    most_basic_model = LogisticRegression()

    baseline_scores = cross_val_score(most_basic_model, X, y, cv=5, scoring="roc_auc")
    baseline_score = np.mean(baseline_scores)
    print("baseline_score " + str(baseline_score))

    dict_scores["LogisticRegressionBaseline"] = baseline_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    p_oos = svc.predict_proba(X_test)
    svm_score = roc_auc_score(y_test, p_oos)
    print("svm score " + str(svm_score))
    dict_scores["SVM"] = svm_score

    with open(os.path.join(config.DATA_FOLDER, "additional_scores.json"), "w") as write_file:
        json.dump(dict_scores, write_file, indent=4)
