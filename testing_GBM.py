import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt 
import GBM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import csv

stocks = ["AAPL", "IBM", "TSLA", "MSFT", "FB", "GOOGL", "PG", "JPM", "NFLX", "INTC", "ADBE", "JNJ", "GS", "MS", "NDAQ", "GM"]


adj_close = dict()
adj_close2 = dict()
results_GBM = dict()

start_training_year = 2015
start_training_month = 1
start_training_day = 2

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


time_step = 10

RMSE = dict()
MAE = dict()
for stock in stocks:
	print(stock)
	adj_close[stock] = pd.read_csv("data/{}.csv".format(stock)).drop(["High", "Low", "Volume","Open","Close"], axis=1)
	
	So = adj_close[stock].loc[ adj_close[stock]["Date"] == start_test_date ].iloc[0]["Adj Close"]

	mask = (adj_close[stock]['Date'] >= start_training_date) & (adj_close[stock]['Date'] <= end_training_date)
	S_historical = adj_close[stock][mask]

	results_GBM[stock] = GBM.avg_gbm(100000, S_historical,So,start_training_date,end_training_date,start_test_date, end_test_date)
	mask = (adj_close[stock]['Date'] >= start_test_date) & (adj_close[stock]['Date'] <= end_test_date)
	S_test = adj_close[stock].loc[mask]["Adj Close"].to_numpy()

	RMSE[stock] = math.sqrt(mean_squared_error(S_test,results_GBM[stock]))
	MAE[stock] = mean_absolute_error(S_test,results_GBM[stock])


with open('GBM_results_lookback_100K_16stocks_2.csv','w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Stocks", "RMSE test", "MAE test"])
	for stock in stocks:
		writer.writerow([stock, RMSE[stock], MAE[stock] ])

	
	