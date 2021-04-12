import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Parameter Definitions

# So    :   initial stock price
# dt    :   time increment -> a day in our case
# T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
# N     :   number of time points in the prediction time horizon -> T/dt
# t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
# mu    :   mean of historical daily returns
# sigma :   standard deviation of historical daily returns
# b     :   array for brownian increments
# W     :   array for brownian path


def gbm(scen_size,S_historical, So,start_training_date,end_training_date,start_test_date, end_test_date):
	dt = 1
	T = pd.date_range(start = pd.to_datetime(start_test_date, 
              format = "%Y-%m-%d") , 
              end = pd.to_datetime(end_test_date, 
              format = "%Y-%m-%d")).to_series(
              ).map(lambda x: 1 if x.isoweekday() in range(1,6) else 0).sum()

	holidays = get_holidays(pd.date_range(start = end_training_date, 
	                end = end_test_date, freq = 'D').map(lambda x:
	                x if x.isoweekday() in range(1, 6) else np.nan).dropna())
	T = T - len(holidays)

	N = T / dt 

	t = np.arange(1, int(N) + 1)

	returns = (S_historical.loc[1:, 'Adj Close'] - S_historical.shift(1).loc[1:, 'Adj Close']) / S_historical.shift(1).loc[1:, 'Adj Close']

	mu = np.mean(returns)

	sigma = np.std(returns)

	b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}

	W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

	drift = (mu - 0.5 * sigma**2) * t

	diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

	#prediction
	S_t = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)]) 

	return S_t

def avg_gbm(scen_size,S_historical, So,start_training_date,end_training_date,start_test_date, end_test_date):
	return np.mean(gbm(scen_size,S_historical, So,start_training_date,end_training_date,start_test_date, end_test_date),axis=0)


def get_holidays(dates):
	#There are two holidays between Jan 2 and Feb 28: Martin Luther King Jr. Day and President's Day
	holidays = []
	for date in dates:
		if (date.month == 2 and date.day == 18) or (date.month == 1 and date.day == 21):
			holidays.append(date)


	return holidays


