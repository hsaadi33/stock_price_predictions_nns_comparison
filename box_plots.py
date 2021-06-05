import keras
import tensorflow as tf
import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
import LSTM
import TCN
import csv
import seaborn as sns
from scipy import stats
import operator as op

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
models = ["lll","llt","ltl","ltt","tll","tlt","ttl","ttt"]
all_models = ["LSTM", "TCN", "LLL","LLT","LTL","LTT","TLL","TLT","TTL","TTT"]

lstm_model = pd.read_csv("lstm_results_300_pred40_days_new.csv").set_index("Stocks")
tcn_model = pd.read_csv("tcn_results_10_timestep_pred40_days_new.csv").set_index("Stocks")

decomp_models_rmse = pd.read_csv("decomp3_results_pred40_days_RMSE_2_new.csv").set_index("Stocks")
decomp_models_mae = pd.read_csv("decomp3_results_pred40_days_MAE_2_new.csv").set_index("Stocks")

RMSE = dict()
MAE = dict()

RMSE["lstm"] = list(pd.read_csv("lstm_results_300_pred40_days_new.csv")["RMSE test"])
MAE["lstm"] = list(pd.read_csv("lstm_results_300_pred40_days_new.csv")["MAE test"])

RMSE["tcn"] = list(pd.read_csv("tcn_results_10_timestep_pred40_days_new.csv")["RMSE test"])
MAE["tcn"] = list(pd.read_csv("tcn_results_10_timestep_pred40_days_new.csv")["MAE test"])

for model in models:
	RMSE[model] = list(pd.read_csv("decomp3_results_pred40_days_RMSE_2_new.csv")[model])
	MAE[model] = list(pd.read_csv("decomp3_results_pred40_days_MAE_2_new.csv")[model])



RMSE["GBM"] = list(pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["RMSE test"])
MAE["GBM"] = list(pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["MAE test"])




#labels, data = [*zip(*RMSE.items())]  # 'transpose' items to parallel key, value lists
labels, data = [*zip(*MAE.items())]  # 'transpose' items to parallel key, value lists

ax = sns.boxplot(data=data, width=.18, showmeans=True,meanprops={"marker": "o", "markeredgecolor": "yellow","markersize": "5"}, whis=[0,100])
plt.xticks(range(0, len(labels)), labels)
#plt.xticks(plt.xticks()[0], list(RMSE.keys()))
#plt.title("RMSE")
ax.set_xlabel("Model")
#ax.set_ylabel("RMSE")
ax.set_ylabel("MAE")
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')
plt.minorticks_on()

plt.show()