from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)



def lstm(F,time_step,num_neurons, start_training_date,end_training_date,start_test_date,end_test_date):
	'''Inputs:
			F: stock adj_prices
			time_step: lag
			num_neurons: number of neurons in LSTM
			start_training_date, end_training_date,start_test_date,end_test_date
		Outputs:
			root mean_squared_error on train
			root mean_squared_error on test
			mean_absolute_error on train
			mean_absolute_error on test'''

	mask = (F['Date'] >= start_training_date) & (F['Date'] <= end_training_date)

	training_size = int(len(F.loc[mask]))
	test_size = len(F)-training_size


	training_set = F.iloc[:training_size,1:].values
	test_set = F.iloc[training_size:,1:].values

	scaler = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = scaler.fit_transform(training_set)
	test_set_scaled = scaler.transform(test_set)
	X_train, y_train = create_dataset(training_set_scaled, time_step)
	X_test, y_test = create_dataset(test_set_scaled, time_step)


	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


	### Create One Layer of LSTM model
	model=Sequential()
	model.add(LSTM(num_neurons,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error',optimizer='adam')

	model.fit(X_train,y_train,epochs=100,batch_size=64,verbose=1, shuffle=False)

	### Prediction and check performance metrics
	y_train_predict = model.predict(X_train)
	y_test_predict = model.predict(X_test)


	##Transformback to original form
	y_train_predict = scaler.inverse_transform(y_train_predict)
	y_test_predict = scaler.inverse_transform(y_test_predict)
	y_train = scaler.inverse_transform(y_train.reshape(-1,1))
	y_test = scaler.inverse_transform(y_test.reshape(-1,1))

	### Calculate RMSE and MAE performance metrics
	return [math.sqrt(mean_squared_error(y_train,y_train_predict)), math.sqrt(mean_squared_error(y_test,y_test_predict)),
		mean_absolute_error(y_train,y_train_predict),mean_absolute_error(y_test,y_test_predict)]



def lstm_decomp3(F,time_step,num_neurons, start_training_date,end_training_date,start_test_date,end_test_date):
	'''Inputs:
			F: stock adj_prices
			time_step: lag
			num_neurons: number of neurons in LSTM
			start_training_date, end_training_date,start_test_date,end_test_date
		Outputs:
			root mean_squared_error on train
			root mean_squared_error on test
			mean_absolute_error on train
			mean_absolute_error on test'''

	mask = (F['Date'] >= start_training_date) & (F['Date'] <= end_training_date)

	training_size = int(len(F.loc[mask]))
	test_size = len(F)-training_size


	training_set = F.iloc[:training_size,1:].values
	test_set = F.iloc[training_size:,1:].values

	scaler = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = scaler.fit_transform(training_set)
	test_set_scaled = scaler.transform(test_set)
	X_train, y_train = create_dataset(training_set_scaled, time_step)
	X_test, y_test = create_dataset(test_set_scaled, time_step)


	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


	### Create One Layer of LSTM model
	model=Sequential()
	model.add(LSTM(num_neurons,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error',optimizer='adam')

	model.fit(X_train,y_train,epochs=100,batch_size=64,verbose=1, shuffle=False)

	### Prediction and check performance metrics
	y_train_predict = model.predict(X_train)
	y_test_predict = model.predict(X_test)


	##Transformback to original form
	y_train_predict = scaler.inverse_transform(y_train_predict)
	y_test_predict = scaler.inverse_transform(y_test_predict)
	y_train = scaler.inverse_transform(y_train.reshape(-1,1))
	y_test = scaler.inverse_transform(y_test.reshape(-1,1))

	
	return [y_train, y_train_predict,y_test,y_test_predict]