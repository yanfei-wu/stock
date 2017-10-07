# Stock Market Analysis and Modeling  

## Introduction 
This project analyzes historical stock data retrived from Google Finance and performs a series of analyses and modeling to   

* understand and compare stock performance 
* build predictive models to make recommendations on trading actions   
* design and evaluate portfolio holdings and trading strategies  

#### Part 1. Five-Year S&P 500 Stock Analysis ([`SP500_Stock_Analysis.ipynb`](https://github.com/yanfei-wu/stock/blob/master/SP500_Stock_Analysis.ipynb))  

In this part, common performance metrics including returns, alpha, beta, sharpe ratio, and moving averages are analyzed for stocks contained in the S&P 500 index. The price data of SPY, i.e., SPDR S&P 500 trust ETF (designed to track the S&P 500 stock market index), is used as reference for performance comparisons and also serves as benchmark for calculating alpha and beta. In addition to individual stocks, the averaged performance of different stock sectors in S&P 500 are also compared. The best and worst performing stocks/sectors are found for specific sector.  

#### Part 2. Machine Learning of Stock Market (`model/`)   

As an initial attempt, a simple classification model is built to help make trading decisions. The stock data used is the historical price for holdings in the PowerShares QQQ ETF. Based on the price change (averaged 1-day to 5-day price changes) of the target stock, labels are generated (1 - buy, 0 - hold, and -1 - sell) for the classification task. The daily returns for all the rest of the holdings in the ETF are used as the features. The model builds upon the `VotingClassifier` consisting of support vector machines, logistic regression, and random forest.   

*Next step*: build RNN models to model the stock price data

#### Part 3. Portfolio Holdings and Trading Strategies (ongoing)  

*** 


## Data, Libraries, and Functions  

The stock data used in this project were obtained from Google Finance. A customer script (`get_data.py`) was used to automate the process and compile all the data into one .csv file.   

The project runs in **Python 3.5** with the following Python libraries:  

- Numpy 
- Pandas 
- Matplotlib 
- seaborn 
- Scikit-Learn 

Some helper functions for the project can be find in the repo, including plotting candlestick chart with moving averages (`vis.py`), manipulating data and calculating performance metrics (`helper.py`).  


