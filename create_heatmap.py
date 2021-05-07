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

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
models = ["lll","llt","ltl","ltt","tll","tlt","ttl","ttt"]
all_models = ["LSTM", "TCN", "LLL","LLT","LTL","LTT","TLL","TLT","TTL","TTT"]

lstm_model = pd.read_csv("lstm_results_300_pred40_days_2.csv").set_index("Stocks")
tcn_model = pd.read_csv("tcn_results_10_timestep_pred40_days_2.csv").set_index("Stocks")

decomp_models_rmse = pd.read_csv("decomp3_results_pred40_days_RMSE_2.csv").set_index("Stocks")
decomp_models_mae = pd.read_csv("decomp3_results_pred40_days_MAE_2.csv").set_index("Stocks")


result = []

'''
for stock in stocks:
	temp = [ lstm_model["RMSE test"][stock], tcn_model["RMSE test"][stock], decomp_models_rmse["lll"][stock]
		 , decomp_models_rmse["llt"][stock] , decomp_models_rmse["ltl"][stock], decomp_models_rmse["ltt"][stock] , decomp_models_rmse["tll"][stock]
		   , decomp_models_rmse["tlt"][stock]  , decomp_models_rmse["ttl"][stock], decomp_models_rmse["ttt"][stock]]
	result.append(temp)'''

for stock in stocks:
	temp = [ lstm_model["MAE test"][stock], tcn_model["MAE test"][stock], decomp_models_mae["lll"][stock]
		 , decomp_models_mae["llt"][stock] , decomp_models_mae["ltl"][stock], decomp_models_mae["ltt"][stock] , decomp_models_mae["tll"][stock]
		   , decomp_models_mae["tlt"][stock]  , decomp_models_mae["ttl"][stock], decomp_models_mae["ttt"][stock]]
	result.append(temp)

result2 = [ -stats.zscore(np.array(row)) for row in result]


ax = sns.heatmap(result2,xticklabels=all_models , yticklabels=stocks,annot=result, cmap="RdYlGn", cbar=False, fmt=".3f")
#plt.title("RMSE values with Z-score colors per row")
plt.title("MAE values with Z-score colors per row")
ax.set_xlabel("Model")
ax.set_ylabel("Stock")

plt.show()