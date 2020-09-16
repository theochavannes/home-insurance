import pandas as pd
from home_insurance.config import config
from home_insurance.constants import *
import numpy as np

print(config.DATA_PATH)
data = pd.read_csv(config.DATA_PATH)
data = data.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
data = data.drop(["PAYMENT_FREQUENCY"], axis=1)
data = data.dropna(
    subset=set(data.columns).difference(["QUOTE_DATE"]),
    how="all")
data["POL_STATUS"] = (data["POL_STATUS"]=="Lapsed").astype(int)
cat_corr_dict = {}
for col in BINARY_COLS:
    cat_corr = data.groupby([col])['POL_STATUS'].value_counts().rename("cat_corr_"+col).reset_index()
    cat_corr = cat_corr[cat_corr["POL_STATUS"]==1].drop("POL_STATUS",axis=1).set_index(col)
    cat_corr = cat_corr / cat_corr.sum()
    cat_corr_dict[col] = cat_corr.max().iloc[0] - cat_corr.min().iloc[0]

data_cat_corr = pd.Series(cat_corr_dict).sort_values(ascending=False)