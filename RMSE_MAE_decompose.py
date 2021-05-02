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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import ast

RMSE = dict()
MAE = dict()

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
models = dict()

df = pd.read_csv("decomp3_results_pred40_days_2.csv").set_index("Stocks")

for stock in stocks:
	df["lstm_trend"][stock] = ast.literal_eval(df["lstm_trend"][stock])
	df["lstm_seasonal"][stock] = ast.literal_eval(df["lstm_seasonal"][stock])
	df["lstm_resid"][stock] = ast.literal_eval(df["lstm_resid"][stock])
	df["tcn_trend"][stock] = ast.literal_eval(df["tcn_trend"][stock])
	df["tcn_seasonal"][stock] = ast.literal_eval(df["tcn_seasonal"][stock])
	df["tcn_resid"][stock] = ast.literal_eval(df["tcn_resid"][stock])
	df["y_test"][stock] = ast.literal_eval(df["y_test"][stock])


models["lll"] = {stock: [df["lstm_trend"][stock][i] + df["lstm_seasonal"][stock][i] + df["lstm_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["llt"] = {stock: [df["lstm_trend"][stock][i] + df["lstm_seasonal"][stock][i] + df["tcn_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["ltl"] = {stock: [df["lstm_trend"][stock][i] + df["tcn_seasonal"][stock][i] + df["lstm_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["ltt"] = {stock: [df["lstm_trend"][stock][i] + df["tcn_seasonal"][stock][i] + df["tcn_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["tll"] = {stock: [df["tcn_trend"][stock][i] + df["lstm_seasonal"][stock][i] + df["lstm_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["tlt"] = {stock: [df["tcn_trend"][stock][i] + df["lstm_seasonal"][stock][i] + df["tcn_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["ttl"] = {stock: [df["tcn_trend"][stock][i] + df["tcn_seasonal"][stock][i] + df["lstm_resid"][stock][i]  for i in range(40)] for stock in stocks}
models["ttt"] = {stock: [df["tcn_trend"][stock][i] + df["tcn_seasonal"][stock][i] + df["tcn_resid"][stock][i]  for i in range(40)] for stock in stocks}
test = {stock: df["y_test"][stock] for stock in stocks}


results = dict()

for stock in stocks:
	RMSE[stock] = [math.sqrt(mean_squared_error(test[stock],models["lll"][stock])),math.sqrt(mean_squared_error(test[stock],models["llt"][stock])),
					math.sqrt(mean_squared_error(test[stock],models["ltl"][stock])),math.sqrt(mean_squared_error(test[stock],models["ltt"][stock])),
					math.sqrt(mean_squared_error(test[stock],models["tll"][stock])),math.sqrt(mean_squared_error(test[stock],models["tlt"][stock])),
					math.sqrt(mean_squared_error(test[stock],models["ttl"][stock])),math.sqrt(mean_squared_error(test[stock],models["ttt"][stock]))]

	MAE[stock] = [mean_absolute_error(test[stock],models["lll"][stock]), mean_absolute_error(test[stock],models["llt"][stock]),
					mean_absolute_error(test[stock],models["ltl"][stock]), mean_absolute_error(test[stock],models["ltt"][stock]),
					mean_absolute_error(test[stock],models["tll"][stock]), mean_absolute_error(test[stock],models["tlt"][stock]),
					mean_absolute_error(test[stock],models["ttl"][stock]), mean_absolute_error(test[stock],models["ttt"][stock]) ]


with open('decomp3_results_pred40_days_RMSE_2.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "lll", "llt", "ltl", "ltt","tll","tlt","ttl","ttt"])
	for stock in stocks:
		writer.writerow([stock, RMSE[stock][0],RMSE[stock][1], RMSE[stock][2], RMSE[stock][3],RMSE[stock][4],RMSE[stock][5],RMSE[stock][6],RMSE[stock][7]])

with open('decomp3_results_pred40_days_MAE_2.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "lll", "llt", "ltl", "ltt","tll","tlt","ttl","ttt"])
	for stock in stocks:
		writer.writerow([stock, MAE[stock][0],MAE[stock][1], MAE[stock][2], MAE[stock][3],MAE[stock][4],MAE[stock][5],MAE[stock][6],MAE[stock][7]])

'''
RMSE_res = dict()
MAE_res = dict()

for stock in stocks:
	rmse_temp = [RMSE[stock][0],RMSE[stock][1], RMSE[stock][2], RMSE[stock][3],RMSE[stock][4],RMSE[stock][5],RMSE[stock][6],RMSE[stock][7]]
	mae_temp = [MAE[stock][0],MAE[stock][1], MAE[stock][2], MAE[stock][3],MAE[stock][4],MAE[stock][5],MAE[stock][6],MAE[stock][7]]
	RMSE_res[stock] = rmse_temp.index(min(rmse_temp))
	MAE_res[stock] = mae_temp.index(min(mae_temp))

RMSE_freq = dict()
MAE_freq = dict()'''



