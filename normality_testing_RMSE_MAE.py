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
from scipy.stats import kstest, norm
from scipy import stats
import statsmodels.api as sm
from scipy.stats import norm
import pylab

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
models = ["lll","llt","ltl","ltt","tll","tlt","ttl","ttt"]
all_models = ["LSTM", "TCN", "LLL","LLT","LTL","LTT","TLL","TLT","TTL","TTT"]

lstm_model = pd.read_csv("lstm_results_300_pred40_days_2.csv").set_index("Stocks")
tcn_model = pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv").set_index("Stocks")

decomp_models_rmse = pd.read_csv("decomp3_results_pred40_days_RMSE_2.csv").set_index("Stocks")
decomp_models_mae = pd.read_csv("decomp3_results_pred40_days_MAE_2.csv").set_index("Stocks")

RMSE = dict()
MAE = dict()

RMSE["lstm"] = pd.read_csv("lstm_results_300_pred40_days_2.csv")["RMSE test"]
MAE["lstm"] = pd.read_csv("lstm_results_300_pred40_days_2.csv")["MAE test"]

RMSE["tcn"] = pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv")["RMSE test"]
MAE["tcn"] = pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv")["MAE test"]

for model in models:
	RMSE[model] = pd.read_csv("decomp3_results_pred40_days_RMSE_2.csv")[model]
	MAE[model] = pd.read_csv("decomp3_results_pred40_days_MAE_2.csv")[model]



RMSE["GBM"] = pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["RMSE test"]
MAE["GBM"] = pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["MAE test"]


print("KS Normality tests for RMSE values")
for key in list(RMSE.keys()):
	ks_statistic, p_value = kstest(RMSE[key], 'norm')
	print(ks_statistic, p_value)

print("KS Normality tests for MAE values")
for key in list(MAE.keys()):
	ks_statistic, p_value = kstest(MAE[key], 'norm')
	print(ks_statistic, p_value)

print("Shapiro Normality tests for RMSE values")
for key in list(RMSE.keys()):
	print(stats.shapiro(RMSE[key]))


print("Shapiro Normality tests for MAE values")
for key in list(MAE.keys()):
	print(stats.shapiro(MAE[key]))

fig, axes = plt.subplots(4, 3, figsize=(6, 4))
ax = axes.flatten()
i = 0
for key in list(RMSE.keys()):
	sm.qqplot(RMSE[key], line='45',ax=ax[i])
	i += 1

fig, axes = plt.subplots(4, 3, figsize=(6, 4))
ax = axes.flatten()
i = 0
for key in list(MAE.keys()):
	sm.qqplot(MAE[key], line='45',ax=ax[i])
	i += 1

pylab.show()


