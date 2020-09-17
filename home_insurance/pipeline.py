"""
This file contains the machine learning pipeline that will be used by the model.
"""
from aikit.pipeline import GraphPipeline
from aikit.transformers import ColumnsSelector
from aikit.transformers import NumericalEncoder, NumImputer
from aikit.transformers.base import PassThrough
from lightgbm.sklearn import LGBMClassifier

from home_insurance.core.transformers import *


columns_selector = ColumnsSelector(columns_to_drop=None,
                                   columns_to_use=['RISK_RATED_AREA_B', 'SUM_INSURED_BUILDINGS', 'NCD_GRANTED_YEARS_B',
                                                   'RISK_RATED_AREA_C', 'SUM_INSURED_CONTENTS', 'NCD_GRANTED_YEARS_C',
                                                   'SPEC_SUM_INSURED', 'SPEC_ITEM_PREM', 'UNSPEC_HRP_PREM', 'BEDROOMS',
                                                   'ROOF_CONSTRUCTION', 'WALL_CONSTRUCTION', 'LISTED', 'MAX_DAYS_UNOCC',
                                                   'OWNERSHIP_TYPE', 'PAYING_GUESTS', 'PROP_TYPE', 'YEARBUILT',
                                                   'MTA_FAP', 'MTA_APRP', 'LAST_ANN_PREM_GROSS', 'QUOTE_DATE_day',
                                                   'QUOTE_DATE_month', 'QUOTE_DATE_year', 'QUOTE_DATE_weekday',
                                                   'QUOTE_DATE_week', 'COVER_START_day', 'COVER_START_month',
                                                   'COVER_START_year', 'COVER_START_weekday', 'COVER_START_week',
                                                   'P1_DOB_day', 'P1_DOB_month', 'P1_DOB_year', 'P1_DOB_weekday',
                                                   'P1_DOB_week', 'MTA_DATE_day', 'MTA_DATE_month', 'MTA_DATE_year',
                                                   'MTA_DATE_weekday', 'MTA_DATE_week', 'TOTAL_SUM']
                                   ,
                                   raise_if_shape_differs=True,
                                   regex_match=False)

imputer = NumImputer(add_is_null=True, allow_unseen_null=True, columns_to_use='all',
                     drop_unused_columns=True, drop_used_columns=True, fix_value=0,
                     regex_match=False, strategy='mean')

numerical_encoder = NumericalEncoder(
    columns_to_use=['CLAIM3YEARS', 'P1_EMP_STATUS', 'P1_PT_EMP_STATUS', 'BUS_USE', 'CLERICAL', 'AD_BUILDINGS',
                    'AD_CONTENTS', 'CONTENTS_COVER', 'BUILDINGS_COVER', 'P1_MAR_STATUS', 'P1_POLICY_REFUSED', 'P1_SEX',
                    'APPR_ALARM', 'APPR_LOCKS', 'FLOODING', 'NEIGH_WATCH', 'OCC_STATUS', 'SAFE_INSTALLED',
                    'SEC_DISC_REQ', 'SUBSIDENCE', 'PAYMENT_METHOD', 'LEGAL_ADDON_PRE_REN', 'LEGAL_ADDON_POST_REN',
                    'HOME_EM_ADDON_PRE_REN', 'HOME_EM_ADDON_POST_REN', 'GARDEN_ADDON_PRE_REN', 'GARDEN_ADDON_POST_REN',
                    'KEYCARE_ADDON_PRE_REN', 'KEYCARE_ADDON_POST_REN', 'HP1_ADDON_PRE_REN', 'HP1_ADDON_POST_REN',
                    'HP2_ADDON_PRE_REN', 'HP2_ADDON_POST_REN', 'HP3_ADDON_PRE_REN', 'HP3_ADDON_POST_REN', 'MTA_FLAG'],
    desired_output_type='DataFrame', drop_unused_columns=True,
    drop_used_columns=True, encoding_type='dummy',
    max_cum_proba=0.95, max_modalities_number=100,
    max_na_percentage=0.05, min_modalities_number=20,
    min_nb_observations=10, regex_match=False)

binary_columns_cleaner = BinaryColumnsCleaner()

# this one does nothing but is used to use the pipeline without the classifier (for shap):
pass_through = PassThrough()
classifier = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                            importance_type='split', learning_rate=0.1, max_depth=-1,
                            min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                            n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                            reg_alpha=0.0, reg_lambda=0.0, silent=True,
                            subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

pipeline = GraphPipeline(edges=[("ColumnsSelector", "NumImputer"),
                                ("NumericalEncoder", "NumImputer", "BinaryColumnsCleaner", "PassThrough",
                                 "LGBMClassifier")],
                         models={"ColumnsSelector": columns_selector,
                                 "NumericalEncoder": numerical_encoder,
                                 "NumImputer": imputer,
                                 "BinaryColumnsCleaner": binary_columns_cleaner,
                                 "PassThrough": pass_through,
                                 "LGBMClassifier": classifier})
