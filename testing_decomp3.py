from keras.preprocessing import sequence
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
import datetime as dt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import stat_decompose

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "PYPL", "ADBE", "JNJ", "GS", "HPE", "MS", "NDAQ", "GM"]
#stocks = ["AAPL"]

adj_close = dict()
results_lstm = dict()
results_tcn = dict()
results_e = dict()

start_training_year = 2015
start_training_month = 1
start_training_day = 9

start_training_year2 = 2015
start_training_month2 = 1
start_training_day2 = 2

end_training_year = 2018
end_training_month = 12
end_training_day = 31

start_test_year = 2019
start_test_month = 1
start_test_day = 2

end_test_year = 2019
end_test_month = 2
end_test_day = 28


start_training_date = "{}-{:02d}-{:02d}".format(start_training_year,start_training_month,start_training_day)
end_training_date = "{}-{:02d}-{:02d}".format(end_training_year,end_training_month,end_training_day)
start_test_date = "{}-{:02d}-{:02d}".format(start_test_year,start_test_month,start_test_day)
end_test_date = "{}-{:02d}-{:02d}".format(end_test_year,end_test_month,end_test_day)

start_training_date2 = "{}-{:02d}-{:02d}".format(start_training_year2,start_training_month2,start_training_day2)

time_step = 10

trend_lstm = dict()
seasonal_lstm = dict()
resid_lstm = dict()

trend_tcn = dict()
seasonal_tcn = dict()
resid_tcn = dict()

y_test = dict()

for stock in stocks:
	adj_close[stock] = pd.read_csv("data2/{}.csv".format(stock)).drop(["High", "Low", "Volume","Open","Close"], axis=1)
	adj_close_trend, adj_close_seasonal, adj_close_resid = stat_decompose.decompose(adj_close[stock])


	trend_lstm[stock] = list(np.concatenate(LSTM.lstm_decomp3(
		adj_close_trend,
		time_step,
		100, 
		start_training_date,
		end_training_date,
		start_test_date,
		end_test_date)[-1]))

	seasonal_lstm[stock] = list(np.concatenate(LSTM.lstm_decomp3(
		adj_close_seasonal,
		time_step,
		100, 
		start_training_date,
		end_training_date,
		start_test_date,end_test_date)[-1]))

	resid_lstm[stock] = list(np.concatenate(LSTM.lstm_decomp3(
		adj_close_resid,
		time_step,
		100, 
		start_training_date,
		end_training_date,
		start_test_date,
		end_test_date)[-1]))


	trend_tcn[stock] = list(np.concatenate(TCN.tcn_decomp3(
		adj_close_trend,
		time_step, 
		start_training_date,
		end_training_date,
		start_test_date,
		end_test_date)[-1]))
	seasonal_tcn[stock] = list(np.concatenate(TCN.tcn_decomp3(
		adj_close_seasonal,
		time_step, 
		start_training_date,
		end_training_date,
		start_test_date,
		end_test_date)[-1]))
	resid_tcn[stock] = list(np.concatenate(TCN.tcn_decomp3(
		adj_close_resid,
		time_step, 
		start_training_date,
		end_training_date,
		start_test_date,
		end_test_date)[-1]))	
		
	F = adj_close[stock]
	mask = (F['Date'] >= start_training_date2) & (F['Date'] <= end_training_date)
	training_size = int(len(F.loc[mask]))
	F = F["Adj Close"]
	F_series = pd.Series(F)
	y_test[stock] = list(F_series[training_size:len(F_series)-5])




with open('decomp3_results_pred40_days.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "lstm_trend", "lstm_seasonal", "lstm_resid", "tcn_trend","tcn_seasonal","tcn_resid","y_test"])
	for stock in stocks:
		writer.writerow([stock, trend_lstm[stock], seasonal_lstm[stock], resid_lstm[stock], trend_tcn[stock], seasonal_tcn[stock], resid_tcn[stock],y_test[stock] ])

