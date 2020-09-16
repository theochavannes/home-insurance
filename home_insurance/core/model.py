from home_insurance.constants import *
from home_insurance.config import config
from home_insurance.pipeline import pipeline

import logging
import os
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
np.random.seed(123)

class Predictor:
    def __init__(self):
        self.model_id = None


    def get_name(self, ext=None):
        name = "home_insurance_{}".format(self.model_id)
        if ext is not None:
            name = name + "." + ext
        return name

    def train(self, data):
        self._initialize()
        data = pd.read_csv(config.DATA_PATH)
        data = data.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
        data = data.drop(["PAYMENT_FREQUENCY"], axis=1)
        data = data.dropna(
            subset=set(data.columns).difference(["QUOTE_DATE"]),
            how="all")

        #data = data.drop(DROP_LOOK_AT, axis=1)
        for col in DATES_COLS:
            data[col] = pd.to_datetime(data[col], errors="coerce")

        data = self._create_dates_features(data, DATES_COLS, True)
        data = self._create_computation_features(data)
        data.to_csv(os.path.join(config.DATA_FOLDER, "data_for_automl.csv"), index=False)
        #data = self._clean_binary_features(data)

        X, y = data.drop(["POL_STATUS"], axis=1), (data["POL_STATUS"]=="Lapsed").astype(int)
        cv = ShuffleSplit(n_splits=5, test_size=0.3)
        self.pipeline = pipeline

        self.scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="roc_auc")
        logging.info("Cross val scores: {}".format(self.scores))

        self.pipeline.fit(X, y)

        self.shap_values = self.get_shap_values(X)
        self._make_shap_plots(X)

    def predict(self, X, top_n=5, round_number=2):
        for col in DATES_COLS:
            X[col] = pd.to_datetime(X[col], errors="coerce")

        X = self._create_dates_features(X, DATES_COLS, True)
        X = self._create_computation_features(X)

        predictions = self.pipeline.predict_proba(X)
        shap_values = list(self.get_shap_values(X))
        predictions = list(predictions)
        if round is not None:
            shap_values = np.round(shap_values, round_number)

        features_influence = pd.DataFrame(shap_values, columns=self._tree_columns)

        results = []
        for i, pred in enumerate(predictions):
            result = {}
            sort = np.abs(features_influence.iloc[i]).sort_values(ascending=False)[:top_n]
            result["explanation"] = features_influence.iloc[i].loc[sort.index].to_dict()
            results.append(result)
        return results

    #
    # def _cross_val_score(self, X, y, cv=None, scoring=None):
    #     if scoring is None:
    #         scoring = roc_auc_score
    #
    #     if cv is None:
    #         cv = ShuffleSplit(n_splits=5, test_size=0.25)
    #
    #     scores = []
    #     for train_index, test_index in cv.split(X):
    #         X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    #         self.pipeline.fit(X, y)
    #
    #         p_oos = self.pipeline.predict_proba(X_test)
    #
    #         scores.append(scoring(X_test, p_oos))
    #
    #     return scores



    def get_shap_values(self, X):
        self.ml_model = self.pipeline.models["LGBMClassifier"]
        self.tree_explainer = shap.TreeExplainer(self.ml_model)
        self.preprocess_pipeline = self.pipeline.get_subpipeline(end_node="PassThrough")
        X_tree = self.preprocess_pipeline.transform(X)
        self._tree_columns = X_tree.columns
        shap_values = self.tree_explainer.shap_values(X_tree)[1]
        return shap_values



    def _initialize(self):
        self.model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_output_path = os.path.join(config.OUTPUT_FOLDER, self.get_name())
        os.mkdir(self.save_output_path)
        self.graph_folder = os.path.join(self.save_output_path, "graphs")
        self.model_folder = os.path.join(self.save_output_path, "model")
        os.mkdir(self.graph_folder)
        os.mkdir(self.model_folder)


    def _create_dates_features(self, data, dates_cols, drop):
        for col in dates_cols:
            day = col + "_" + "day"
            month = col + "_" + "month"
            year = col + "_" + "year"
            week = col + "_" + "week"
            weekday = col + "_" + "weekday"

            data[day] = data[col].dt.day
            data[month] = data[col].dt.month
            data[year] = data[col].dt.year
            data[weekday] = data[col].dt.dayofweek
            data[week] = data[col].dt.week

        if drop:
            data = data.drop(dates_cols, axis=1)

        return data

    def _clean_binary_features(self, data):
        data[BINARY_COLS] = data[BINARY_COLS].apply(lambda col: col.map({"Y":1, "N":0}))
        return data

    def _create_computation_features(self, data):
        data["TOTAL_SUM"] = data["SUM_INSURED_BUILDINGS"] + \
                            data["SUM_INSURED_CONTENTS"] + \
                            data["SPEC_SUM_INSURED"]

        return data

    def _make_shap_plots(self, X):
        X_tree = self.preprocess_pipeline.transform(X)
        shap.summary_plot(self.shap_values, X_tree, show=False, plot_size=(30, 15))
        plt.tight_layout()
        shap_values_path = os.path.join(self.graph_folder, "shap_values_{}".format(self.model_id))
        plt.savefig(shap_values_path)
        plt.clf()

        shap.summary_plot(self.shap_values, X_tree, show=False, plot_size=(30, 15), plot_type="bar")
        plt.tight_layout()
        features_importances_path= os.path.join(self.graph_folder, "importance_{}".format(self.model_id))
        plt.savefig(features_importances_path)
        plt.clf()

        best_indexes = np.abs(self.shap_values).mean(axis=0).argsort()[:50][::-1]
        name_indexes = X_tree.columns[best_indexes]
        dependence_plots_folder = os.path.join(self.graph_folder, "dependence_plots")
        os.mkdir(dependence_plots_folder)

        for feature in name_indexes:
            plt.clf()
            shap.dependence_plot(feature, self.shap_values, X_tree, interaction_index=None, show=False)
            plt.tight_layout()
            dependence_plot_path = os.path.join(dependence_plots_folder, "dependence_plot_{}_{}".format(feature, self.model_id))
            plt.savefig(dependence_plot_path)
            plt.clf()
