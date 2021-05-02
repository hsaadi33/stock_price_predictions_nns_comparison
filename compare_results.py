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

RMSE = dict()
MAE = dict()
stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
models = ["lll","llt","ltl","ltt","tll","tlt","ttl","ttt"]

df1 = pd.read_csv("lstm_results_300_pred40_days_2.csv").set_index("Stocks")
df2 = pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv").set_index("Stocks")

rmse_df = pd.read_csv("decomp3_results_pred40_days_RMSE_2.csv").set_index("Stocks")
mae_df = pd.read_csv("decomp3_results_pred40_days_MAE_2.csv").set_index("Stocks")


for stock in stocks:
	RMSE[stock] = [rmse_df.loc[stock,model] for model in models]+ [df1.loc[stock,"RMSE test"]]+ [df2.loc[stock,"RMSE test"]]
	
	MAE[stock] = [mae_df.loc[stock,model] for model in models]+ [df1.loc[stock,"MAE test"]]+ [df2.loc[stock,"MAE test"]]

RMSE_results = dict()
MAE_results = dict()


for stock in stocks:
	RMSE_results[stock] = RMSE[stock].index(min(RMSE[stock]))
	MAE_results[stock] = MAE[stock].index(min(MAE[stock]))

RMSE_freq = dict()
MAE_freq = dict()

for i in range(10):
	RMSE_freq[i] = 0
	MAE_freq[i] = 0

for i in range(10):
	for stock in stocks:
		if RMSE_results[stock] == i:
			RMSE_freq[i] += 1

		if MAE_results[stock] == i:
			MAE_freq[i] += 1


RMSE_freq["lll"] = RMSE_freq.pop(0)
RMSE_freq["llt"] = RMSE_freq.pop(1)
RMSE_freq["ltl"] = RMSE_freq.pop(2)
RMSE_freq["ltt"] = RMSE_freq.pop(3)
RMSE_freq["tll"] = RMSE_freq.pop(4)
RMSE_freq["tlt"] = RMSE_freq.pop(5)
RMSE_freq["ttl"] = RMSE_freq.pop(6)
RMSE_freq["ttt"] = RMSE_freq.pop(7)
RMSE_freq["lstm"] = RMSE_freq.pop(8)
RMSE_freq["tcn"] = RMSE_freq.pop(9)


MAE_freq["lll"] = MAE_freq.pop(0)
MAE_freq["llt"] = MAE_freq.pop(1)
MAE_freq["ltl"] = MAE_freq.pop(2)
MAE_freq["ltt"] = MAE_freq.pop(3)
MAE_freq["tll"] = MAE_freq.pop(4)
MAE_freq["tlt"] = MAE_freq.pop(5)
MAE_freq["ttl"] = MAE_freq.pop(6)
MAE_freq["ttt"] = MAE_freq.pop(7)
MAE_freq["lstm"] = MAE_freq.pop(8)
MAE_freq["tcn"] = MAE_freq.pop(9)


print(RMSE_freq)
print(MAE_freq)

RMSE_freq2 = dict()
MAE_freq2 =dict()

for stock in stocks:
	if RMSE_results[stock] == 0:
		RMSE_freq2.setdefault("lll",[]).append(stock)
	elif RMSE_results[stock] == 1:
		RMSE_freq2.setdefault("llt",[]).append(stock)
	elif RMSE_results[stock] == 2:
		RMSE_freq2.setdefault("ltl",[]).append(stock)
	elif RMSE_results[stock] == 3:
		RMSE_freq2.setdefault("ltt",[]).append(stock)
	elif RMSE_results[stock] == 4:
		RMSE_freq2.setdefault("tll",[]).append(stock)
	elif RMSE_results[stock] == 5:
		RMSE_freq2.setdefault("tlt",[]).append(stock)
	elif RMSE_results[stock] == 6:
		RMSE_freq2.setdefault("ttl",[]).append(stock)
	elif RMSE_results[stock] == 7:
		RMSE_freq2.setdefault("ttt",[]).append(stock)
	elif RMSE_results[stock] == 8:
		RMSE_freq2.setdefault("lstm",[]).append(stock)
	elif RMSE_results[stock] == 9:
		RMSE_freq2.setdefault("tcn",[]).append(stock)

for stock in stocks:
	if MAE_results[stock] == 0:
		MAE_freq2.setdefault("lll",[]).append(stock)
	elif MAE_results[stock] == 1:
		MAE_freq2.setdefault("llt",[]).append(stock)
	elif MAE_results[stock] == 2:
		MAE_freq2.setdefault("ltl",[]).append(stock)
	elif MAE_results[stock] == 3:
		MAE_freq2.setdefault("ltt",[]).append(stock)
	elif MAE_results[stock] == 4:
		MAE_freq2.setdefault("tll",[]).append(stock)
	elif MAE_results[stock] == 5:
		MAE_freq2.setdefault("tlt",[]).append(stock)
	elif MAE_results[stock] == 6:
		MAE_freq2.setdefault("ttl",[]).append(stock)
	elif MAE_results[stock] == 7:
		MAE_freq2.setdefault("ttt",[]).append(stock)
	elif MAE_results[stock] == 8:
		MAE_freq2.setdefault("lstm",[]).append(stock)
	elif MAE_results[stock] == 9:
		MAE_freq2.setdefault("tcn",[]).append(stock)
print(RMSE_freq2)
print(MAE_freq2)



RMSE2 = dict()
MAE2 = dict()

for model in models:
	RMSE2[model] = np.mean(pd.read_csv("decomp3_results_pred40_days_RMSE_2.csv")[model].to_numpy())
	MAE2[model] = np.mean(pd.read_csv("decomp3_results_pred40_days_MAE_2.csv")[model].to_numpy())

RMSE2["lstm"] = np.mean(pd.read_csv("lstm_results_300_pred40_days_2.csv")["RMSE test"].to_numpy())
MAE2["lstm"] = np.mean(pd.read_csv("lstm_results_300_pred40_days_2.csv")["MAE test"].to_numpy())

RMSE2["tcn"] = np.mean(pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv")["RMSE test"].to_numpy())
MAE2["tcn"] = np.mean(pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv")["MAE test"].to_numpy())

RMSE2["GBM"] = np.mean(pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["RMSE test"].to_numpy())
MAE2["GBM"] = np.mean(pd.read_csv("GBM_results_lookback_100K_16stocks_2.csv")["MAE test"].to_numpy())

print("RMSE2,MAE2")
print(RMSE2)
print(MAE2)

