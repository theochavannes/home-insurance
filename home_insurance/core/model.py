"""
This file contains the class Predictor and all its methods
"""
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import ShuffleSplit, cross_val_score

from home_insurance.config import config
from home_insurance.constants import *
from home_insurance.pipeline import pipeline

np.random.seed(123)


class Predictor:
    def __init__(self):
        self.model_id = None

    def train(self, data):
        """This method trains the Predictor. It does more than a machine learning model. Training the
        predictor creates a folder where it saves all the needed outputs (data for the automl, pickle of itself,
        shap values graphs and dependence plots...). It stores all the important information (scores,
        machine learning pipeline,...)
            Parameters
            ----------
            data: DataFrame
                the input data

            Output
            ------
            None
        """
        self._initialize()
        data = pd.read_csv(config.DATA_PATH)
        data = data.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
        data = data.drop(["PAYMENT_FREQUENCY"], axis=1)
        data = data.dropna(
            subset=set(data.columns).difference(["QUOTE_DATE"]),
            how="all")

        # data = data.drop(DROP_LOOK_AT, axis=1)
        for col in DATES_COLS:
            data[col] = pd.to_datetime(data[col], errors="coerce")

        data = self._create_dates_features(data, DATES_COLS, True)
        data = self._create_computation_features(data)
        data.to_csv(os.path.join(config.DATA_FOLDER, "data_for_automl.csv"), index=False)
        # data = self._clean_binary_features(data)

        X, y = data.drop(["POL_STATUS"], axis=1), (data["POL_STATUS"] == "Lapsed").astype(int)
        cv = ShuffleSplit(n_splits=5, test_size=0.3)
        self.pipeline = pipeline

        self.scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="roc_auc")
        logging.info("Cross val scores: {}".format(self.scores))

        self.pipeline.fit(X, y)

        self.shap_values = self.get_shap_values(X)
        self._make_shap_plots(X)

    def predict(self, data, top_n=5, round_number=2):
        """This method should be used once the trained pickle model is loaded.
            Parameters
            ----------
            data: list
                the input data (records format: list of dictionaries, each element of the list being a data)
            top_n: int
                the top_n most important features for each data of the input (shap values)
            round_number: int
                returning the shap values rounded at round_number

            Output
            ------
            results: list
                a list of dictionaries containing, for element i, the prediction and the shap values
                for the data row number i
        """
        X = pd.DataFrame.from_records(data)
        for col in DATES_COLS:
            X[col] = pd.to_datetime(X[col], errors="coerce")

        X = self._create_dates_features(X, DATES_COLS, True)
        X = self._create_computation_features(X)

        predictions = self.pipeline.predict_proba(X)[:, 1]
        shap_values = list(self.get_shap_values(X))
        predictions = list(predictions)
        if round is not None:
            shap_values = np.round(shap_values, round_number)

        features_influence = pd.DataFrame(shap_values, columns=self._tree_columns)

        results = []
        for i, pred in enumerate(predictions):
            result = {}
            sort = np.abs(features_influence.iloc[i]).sort_values(ascending=False)[:top_n]
            result["confidence"] = predictions[i]
            result["explanation"] = features_influence.iloc[i].loc[sort.index].to_dict()
            results.append(result)
        return results

    def get_name(self, ext=None):
        """
        This method returns the name of package + the model + eventually an extension
        Parameters
            ----------
            ext (Optional): string
                the name of the extension

            Output
            ------
            name: string
                name of package + the model + eventually an extension

        """
        name = "home_insurance_{}".format(self.model_id)
        if ext is not None:
            name = name + "." + ext
        return name

    def get_shap_values(self, X):
        """This method output the shap values, using the pipeline defined in the method train_model
                    Parameters
                    ----------
                    X: DatFrame
                        the input data

                    Output
                    ------
                    shap_values: array
                        The shap values for the data given
                """
        self.ml_model = self.pipeline.models["LGBMClassifier"]
        self.tree_explainer = shap.TreeExplainer(self.ml_model)
        self.preprocess_pipeline = self.pipeline.get_subpipeline(end_node="PassThrough")
        X_tree = self.preprocess_pipeline.transform(X)
        self._tree_columns = X_tree.columns
        shap_values = self.tree_explainer.shap_values(X_tree)[1]
        return shap_values

    def _initialize(self):
        """This method initialize the model. It creates the paths and working directories it needs to store the
        outputs. It also defines the name of the model as the current time.
                    Parameters
                    ----------
                    None

                    Output
                    ------
                    None
                """
        self.model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_output_path = os.path.join(config.OUTPUT_FOLDER, self.get_name())
        if not os.path.exists(self.save_output_path):
            os.mkdir(self.save_output_path)

        self.graph_folder = os.path.join(self.save_output_path, "graphs")
        self.model_folder = os.path.join(self.save_output_path, "model")
        if not os.path.exists(self.graph_folder):
            os.mkdir(self.graph_folder)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

    def _create_dates_features(self, data, dates_cols, drop):
        """This method creates multiple features linked to the dates. It also drops the columns which were used to
        create features (if specified).
        The features created are: day, month, year, weekday, week number.
                    Parameters
                    ----------
                    data: DataFrame
                        the input data
                    dates_cols: list
                        the list of columns of type datetime, from which we want to create our features
                    drop: Boolean
                        Specify if we want to keep dates_cols at the end of the creation of features
                    Output
                    ------
                    data: DataFrame
                        the data with new features created as mentioned
                """
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

    def _create_computation_features(self, data):
        """This method should be used once the trained pickle model is loaded.
        It creates some features which are computed from already existing features
        by doing simple calculations.
                    Parameters
                    ----------
                    data: DataFrame
                        the input data

                    Output
                    ------
                    data: DataFrame
                        the data with newly created features
        """
        data["TOTAL_SUM"] = data["SUM_INSURED_BUILDINGS"] + \
                            data["SUM_INSURED_CONTENTS"] + \
                            data["SPEC_SUM_INSURED"]

        return data

    def _make_shap_plots(self, X):
        """This method creates all the shap plots once the pipeline is fitted. It stores them in the graphs folder.
            Parameters
            ----------
            X: DataFrame
                the input data

            Output
            ------
            None
        """

        # step 1: create shap values plot
        X_tree = self.preprocess_pipeline.transform(X)
        shap.summary_plot(self.shap_values, X_tree, show=False, plot_size=(30, 15))
        plt.tight_layout()
        shap_values_path = os.path.join(self.graph_folder, "shap_values_{}".format(self.model_id))
        plt.savefig(shap_values_path)
        plt.clf()

        # step 2: create features importances barplot
        shap.summary_plot(self.shap_values, X_tree, show=False, plot_size=(30, 15), plot_type="bar")
        plt.tight_layout()
        features_importances_path = os.path.join(self.graph_folder, "importance_{}".format(self.model_id))
        plt.savefig(features_importances_path)
        plt.clf()

        # step 3: create all the dependence plots from the 50 most important features, according to shap
        best_indexes = np.abs(self.shap_values).mean(axis=0).argsort()[-50:][::-1]
        name_indexes = X_tree.columns[best_indexes]
        dependence_plots_folder = os.path.join(self.graph_folder, "dependence_plots")
        if not os.path.exists(dependence_plots_folder):
            os.mkdir(dependence_plots_folder)
            os.mkdir(os.path.join(dependence_plots_folder, "most_important_dependences"))

        for i, feature in enumerate(name_indexes):
            plt.clf()
            shap.dependence_plot(feature, self.shap_values, X_tree, interaction_index=None, show=False)
            plt.tight_layout()
            if i < 10:
                dependence_plot_path = os.path.join(dependence_plots_folder, "most_important_dependences",
                                                    "dependence_plot_{}_{}_{}".format(i + 1, feature, self.model_id))
            else:
                dependence_plot_path = os.path.join(dependence_plots_folder,
                                                    "dependence_plot_{}_{}_{}".format(i + 1, feature, self.model_id))
            plt.savefig(dependence_plot_path)
