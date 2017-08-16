import numpy as np
import pandas as pd
from datetime import datetime

def get_close_price(data, tickers):
    """
    generate a dataframe with the close price for a given list of stocks
    @param data: a dataframe with columns 'Date', 'Name', 'Close' and other optional columns
    @param tickers: a list of stock tiker symbols
    --
    return: a dataframe with datetime index and close prices and close prices 
            of each stock populated in each column        
    """

    close = data[data['Name'] == tickers[0]][['Date']]
    for t in tickers:
        close[t] = data[data['Name'] == t]['Close'].values
    close.set_index(['Date'], inplace=True)
    close.index = pd.to_datetime(close.index)
    
    return close


def calculate_metrics(close, tickers):
	"""
    generate a dataframe with the performance of the matrics
    @param close: a dataframe with the close price of all stocks
    @param tickers: a list of stock tiker symbols
    --
    return: a dataframe with stock ticker as index and performance values
    """

	daily_rf = (1 + 0.01)**(1/252) - 1
	daily_retn = close[tickers].pct_change().dropna()
	daily_retn_adj = daily_retn - daily_rf
	
	# benchmark -- SPY
	ref_daily_retn = daily_retn['SPY'].values
	ref_annual_retn = (close['SPY'][-1] - close['SPY'][0])/close['SPY'][0]
	ref_annual_retn = (ref_annual_retn + 1)**(1/5) - 1
	
	name = []
	total_retn = []
	annual_retn = []
	volatility = []
	avg_daily_retn = []
	alpha = []
	beta = []
	sharpe = []

	for t in tickers:
		name.append(t)
		s_retn = (close[t][-1] - close[t][0])/close[t][0] # simple 5-year return
		a_retn = (s_retn + 1)**(1/5) - 1 # annualized return
		total_retn.append(s_retn)
		annual_retn.append(a_retn)

		volatility.append(daily_retn[t].std())
		avg_daily_retn.append(daily_retn[t].mean())

		sharpe.append(daily_retn_adj[t].mean() / daily_retn_adj[t].std())

		b = np.cov(daily_retn[t].values, ref_daily_retn)[0][1]/np.var(ref_daily_retn) # beta
		beta.append(b)

		a = a_retn - 0.01 - b*(ref_annual_retn - 0.01) # alpha
		alpha.append(a)

	results = pd.DataFrame({'total_retn': total_retn, 
							'annual_retn': annual_retn,
							'volatility': volatility,
							'avg_daily_retn': avg_daily_retn, 
							'alpha': alpha, 
							'beta': beta,
							'sharpe': sharpe
							}, index=name) 

	return results


