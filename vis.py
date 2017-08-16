#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 2017

Author: Yanfei Wu
Helper functions for s&p stock analysis 
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num

def plot_candlestick_ohlc(data, stick='Daily', ticker='AAPL', ma=False, ma_interval=None, 
                            start_date='2012-09-01', end_date='2017-08-14'):
    """
    plot a Japanese candlestick chart for stock data
    @param data: pandas DataFrame object, 
                    with columns 'Name', 'Date', 'Open', 'High', 'Low', 'Close'
    @param stick: string indicating the period of time covered by a single candlestick. 
    @param ticker: string representing the stock ticker symbol 
    @param ma: flag to indicate if to plot moving averages
    @param ma_interval: a list of integers of moving average intervals
    @param start_date: string in the format of 'YYYY-MM-DD' indicating the starting plot range
    @param end_date: string in the format of 'YYYY-MM-DD' indicating the starting plot range
    """

    mondays = WeekdayLocator(MONDAY)    # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')  # e.g., 12
 
    # check ticker input
    if ticker not in data['Name'].unique():
        raise ValueError('Invalid ticker input')

    # check stick input and convert stick argument to integer
    if (type(stick) == str):
        if stick == 'Daily':
            s = 1
        elif stick == 'Weekly':
            s = 7
        elif stick == 'Monthly':
            s = 30
        elif stick == 'Yearly':
            s = 365
        else: 
            raise ValueError('Invalid stick input')

    # subset the data    
    transdata = data.loc[data['Name'] == ticker, ['Date', 'Open', 'High', 'Low', 'Close']]
    transdata.set_index(['Date'], inplace=True)
    transdata.index = pd.to_datetime(transdata.index)
    
    # create new dataframe to be plotted
    transdata['stick'] = [np.floor(i / s) for i in range(len(transdata.index))]
    grouped = transdata.groupby('stick')
    plotdata = pd.DataFrame({'Open': [], 'High': [], 'Low': [], 'Close': []}) 
    for name, group in grouped:
        plotdata = plotdata.append(pd.DataFrame({'Open': group.iloc[0, 0],
                                    'High': max(group.High),
                                    'Low': min(group.Low),
                                    'Close': group.iloc[-1, 3]},
                                    index = [group.index[0]]))
 
    # set plot parameters
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(bottom=0.2)
    if plotdata.index[-1] - plotdata.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True, linestyle=':', linewidth=0.75)
 
    # create the candelstick chart and format the plot
    candlestick_ohlc(ax, list(zip(list(date2num(plotdata.loc[start_date:end_date, :].index.tolist())), 
                                  plotdata.loc[start_date:end_date, 'Open'].tolist(), 
                                  plotdata.loc[start_date:end_date, 'High'].tolist(),
                                  plotdata.loc[start_date:end_date, 'Low'].tolist(), 
                                  plotdata.loc[start_date:end_date, 'Close'].tolist())),
                     colorup='green', colordown='red', width=s*.4)

    # if ma is set to True, calculate moving averages and plot them as lines
    if ma == True:
        if not ma_interval:
            raise ValueError('Invalid moving average intervals input')
        else: 
            ma_cols = []
            for i in ma_interval:
                line_name = str(i)+'d_ma'
                ma_cols.append(line_name)
                plotdata[line_name] = np.round(plotdata["Close"].rolling(window=i, 
                                                                        center=False).mean(), 2)
            plotdata.loc[start_date:end_date, ma_cols].plot(ax=ax, lw=1, grid=True, legend=True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.ylabel('Price')
    plt.title(stick + ' Stock Price of ' + ticker)
 
    plt.show()

