#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 2017
Modified on Wed Aug 16 2017

Author: Yanfei Wu
Get the past 500 S&P 500 stocks data
"""

from bs4 import BeautifulSoup
import requests
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web

def get_ticker_and_sector(url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'):
	""" 
	get the s&p 500 stocks from Wikipedia:
	https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
	
	---
	return: a dictionary with ticker names as keys and sectors as values 
	"""

	r = requests.get(url)
	data = r.text
	soup = BeautifulSoup(data, 'lxml')

	# we only want to parse the first table of this wikipedia page
	table = soup.find('table')

	sp500 = {}
	# loop over the rows and get ticker symbol and sector name
	for tr in table.find_all('tr')[1:]: 
		tds = tr.find_all('td')
		ticker = tds[0].text
		sector = tds[3].text
		sp500[ticker] = sector

	return sp500

def get_stock_data(ticker, start_date, end_date):
	""" get stock data from google with stock ticker, start and end dates """
	data = web.DataReader(ticker, 'google', start_date, end_date)
	return data

if __name__ == '__main__':

	""" get the stock data from the past 5 years """
	# end_date = datetime.now()
	end_date = datetime(2017, 8, 14)
	start_date = datetime(end_date.year - 5, end_date.month , end_date.day)

	sp500 = get_ticker_and_sector()
	sp500['SPY'] = 'SPY' # also include SPY as reference
	print('Total number of tickers (including SPY): {}'.format(len(sp500))) 

	bad_tickers =[]
	for i, (ticker, sector) in enumerate(sp500.items()):
		try:
			stock_df = get_stock_data(ticker, start_date, end_date)
			stock_df['Name'] = ticker
			stock_df['Sector'] = sector
			if stock_df.shape[0] == 0:
				bad_tickers.append(ticker)
			#output_name = ticker + '_data.csv'
			#stock_df.to_csv(output_name)
			if i == 0:
				all_df = stock_df
			else:
				all_df = all_df.append(stock_df)
		except:
			bad_tickers.append(ticker)
	print(bad_tickers)

	all_df.to_csv('./data/all_sp500_data_2.csv')

	""" Write failed queries to a text file """
	if len(bad_tickers) > 0:
		with open('./data/failed_queries_2.txt','w') as outfile:
			for ticker in bad_tickers:
				outfile.write(ticker+'\n')
