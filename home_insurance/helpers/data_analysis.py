import pandas as pd
from home_insurance.config import config
from home_insurance.constants import *
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from datetime import datetime

print(config.DATA_PATH)
data = pd.read_csv(config.DATA_PATH)
data = data.drop(["i", "Police", "CAMPAIGN_DESC"], axis=1)
data = data.drop(["PAYMENT_FREQUENCY"], axis=1)
data = data.dropna(
    subset=set(data.columns).difference(["QUOTE_DATE"]),
    how="all")
data["POL_STATUS"] = (data["POL_STATUS"]=="Lapsed").astype(int)

pol_status_count = (data["POL_STATUS"].value_counts()/data.shape[0]).to_frame()

sns.barplot(x=data['POL_STATUS'].value_counts().index, y=data["POL_STATUS"].value_counts(normalize=True))

if not os.path.exists(config.DATA_ANALYSIS_FOLDER):
    os.mkdir(config.DATA_ANALYSIS_FOLDER)
analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
current_folder = os.path.join(config.DATA_ANALYSIS_FOLDER, analysis_id)
cat_corr_folder = os.path.join(current_folder, "cat_corr")
corr_folder = os.path.join(current_folder, "corr")
os.mkdir(current_folder)
os.mkdir(cat_corr_folder)
os.mkdir(corr_folder)


cat_corr_dict = {}
for col in CATEGORICAL_FEATURES:
    cat_corr = data.groupby([col])['POL_STATUS'].value_counts(normalize=True).rename("cat_corr_"+col).reset_index()
    cat_corr = cat_corr[cat_corr["POL_STATUS"]==1].drop("POL_STATUS",axis=1).set_index(col)
    cat_corr_dict[col] = cat_corr.max().iloc[0] - cat_corr.min().iloc[0]

data_cat_corr = pd.Series(cat_corr_dict).sort_values(ascending=False)

for i, col in enumerate(data_cat_corr.index):
    plt.clf()
    cat_corr = data.groupby([col])['POL_STATUS'].value_counts(normalize=True).rename("cat_corr_" + col).reset_index()
    cat_corr = cat_corr[cat_corr["POL_STATUS"] == 1].drop("POL_STATUS", axis=1).set_index(col)
    sns_plot = sns.barplot(x=cat_corr["cat_corr_"+col].index,
                y=cat_corr["cat_corr_"+col])
    plt.savefig(os.path.join(cat_corr_folder, "cat_corr_{}_{}.png".format(i, col)))

plt.clf()
print(data.loc[data["MTA_FAP"].notna(), "MTA_FAP"] == data.loc[data["MTA_FAP"].notna(), "LAST_ANN_PREM_GROSS"])
##when there is no nans, they contain the same values

plt.figure(figsize=(15, 10))
corr_matrix = data.select_dtypes(include=[np.number]).drop("MTA_FAP", axis=1).corr()
most_correlated=corr_matrix["POL_STATUS"].abs().sort_values(ascending=False).index[:10]
corr_plot = sns.heatmap(corr_matrix.loc[most_correlated, most_correlated], cmap="coolwarm", annot=True)
corr_plot.set_xticklabels(corr_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.savefig(os.path.join(corr_folder, "correlation_matrix.png"))


