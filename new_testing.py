from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
import new_LSTM
import new_TCN
import csv
import datetime as dt 

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]
#stocks = ["AAPL"]


adj_close = dict()
results_lstm = dict()
results_tcn = dict()
results_e = dict()

start_training_year = 2015
start_training_month = 1
start_training_day = 2

#12/31/2018 minus time_step
end_training_year = 2018
end_training_month = 12
end_training_day = 13

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

time_step = 10
for stock in stocks:
	adj_close[stock] = pd.read_csv("data/{}.csv".format(stock)).drop(["High", "Low", "Volume","Open","Close"], axis=1)
	results_lstm[stock] = new_LSTM.lstm(adj_close[stock],time_step,300, start_training_date,end_training_date,start_test_date,end_test_date)
	results_tcn[stock] = new_TCN.tcn(adj_close[stock],time_step, start_training_date,end_training_date,start_test_date,end_test_date)
	

with open('lstm_results_300_pred40_days_new.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "RMSE train", "RMSE test", "MAE train" , "MAE test"])
	for stock in stocks:
		writer.writerow([stock, results_lstm[stock][0], results_lstm[stock][1], results_lstm[stock][2],results_lstm[stock][3]])

with open('tcn_results_10_timestep_pred40_days_new.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "RMSE train", "RMSE test", "MAE train" , "MAE test"])
	for stock in stocks:
		writer.writerow([stock, results_tcn[stock][0], results_tcn[stock][1], results_tcn[stock][2],results_tcn[stock][3]])
		








