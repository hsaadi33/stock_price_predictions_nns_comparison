import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime as dt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import math

def decompose(df):
	df2 = df.set_index("Date", inplace=False)

	res = sm.tsa.seasonal_decompose(df2, model="additive",period = 10,two_sided=False)

	#Moving average in trend method doesn't produce values for the first 10 elements (convolution filter with moving average for the past 10 days)
	trend = res.trend[10:].to_frame().reset_index()
	trend = trend.rename(columns={"trend": "Adj Close"})
	seasonal = res.seasonal[10:].to_frame().reset_index()
	seasonal = seasonal.rename(columns={"seasonal": "Adj Close"})
	resid = res.resid[10:].to_frame().reset_index()
	resid = resid.rename(columns={"resid": "Adj Close"})

	return trend , seasonal, resid

	
	

