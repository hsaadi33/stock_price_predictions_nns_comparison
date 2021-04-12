"To Decompose or not to Decompose: A Comparison between Different Neural Networks and GBM models for Stock Prices Predictions."


In this project there are 9 files:
1. LSTM.py has the code for two types of LSTMs that are mentioned in the report.
2. TCN.py has the code for two TCNs, but they have the same hyperparameters. 
3. GBM.py has the code for simulating Geometric Brownian Motion for prices.
4. stat_decompose.py decomposes the time series into three parts: trend, seasonality, and residual.
5. testing.py tests the LSTM and TCN.
6. testing_GBM.py test the GBM model.
7. testing_decomp3.py writes the predictions of the eight decomposition models in csv files.
8. RMSE_MAE_decompose.py collects the predictions from the csv files that were produced from testing_decomp3, and compares them with the real data.
9. compare_results.py prints the best models that predicted a certain stock and the average RMSE and MAE.



## Report Abstract: 
In this work, 18 stocks are considered to predict the adjusted closing price for the next 40 trading days. The methods that were used to make predictions are: Long Short Term Memory (LSTM), Temporal Convolution Networks (TCN), decomposing the signal and applying different combinations of LSTM and TCN on the decomposed parts, and stochastic process-geometric Brownian motion. The historical data from Yahoo Finance from 2015-2018 was used to build all the models. Finally, the output of each model is compared to the actual adjusted closing price of each stock. The results show that applying a TCN yielded on average the best results and it has the smallest number of parameters among all neural networks models. However, decomposition with a trend, seasonality, and noise compoenents of a time series yielded the best results for some stocks. 



For bugs and questions, contact: saadi.cv4 at gmail.com



If the code and report helps your research, please cite the work with the title and author name as written in the report.
