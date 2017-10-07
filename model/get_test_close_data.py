#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 2017

Author: Yanfei Wu
Get the close price data for stocks in QQQ holdings in the past 8 days
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
import pandas_datareader.data as web

def get_stock_data(ticker, start_date, end_date):
	""" get stock data from google with stock ticker, start and end dates """
	data = web.DataReader(ticker, 'google', start_date, end_date)
	return data

if __name__ == '__main__':

	""" get the stock data from the past 8 days """
	end_date = datetime.now()
	start_date = end_date - timedelta(8)

	# get tickers
	qqq = pd.read_excel('./data/qqq_weight.xlsx') # holdings' weight dataframe
	tickers = qqq.Symbol.unique()
	tickers = np.append(tickers, 'TQQQ') # also include TQQQ 
	tickers = tickers[tickers != 'LILA'] # modified on 9/8/17
	print('Total number of tickers (including TQQQ): {}'.format(len(tickers))) 

	# get close data and join together
	data = pd.DataFrame()
	bad_tickers =[]
	for i, ticker in enumerate(tickers):
		try:
			stock = get_stock_data(ticker, start_date, end_date)[['Close']]
			stock.rename(columns={'Close': ticker}, inplace=True)

			if stock.shape[0] == 0:
				bad_tickers.append(ticker)
			
			if data.empty:
				data = stock
			else:
				data = data.join(stock, how='outer')
		except:
			bad_tickers.append(ticker)

	# write to file
	data.to_csv('./data/test_data.csv')

	""" write failed queries to a text file """
	if len(bad_tickers) > 0:
		print(bad_tickers)
		with open('./data/failed_queries.txt','w') as outfile:
			for ticker in bad_tickers:
				outfile.write(ticker+'\n')