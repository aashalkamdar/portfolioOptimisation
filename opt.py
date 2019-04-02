# Imports
from stockapi import getStockData
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
np.random.seed(0)

def optimize(stocks):
    # stocks is an array of ticker symbols
    stock_dfs = {}

    for ticker in stocks:
        print(ticker)
        stock_dfs[ticker] = getStockData(ticker)
    stock = pd.concat(list(stock_dfs.values()),axis=1)
    stock.columns = list(stock_dfs.keys())
    stock = stock.loc[~(stock==0).any(axis=1)] 
    stock.dropna(inplace=True)

    log_ret = np.log10(stock/stock.shift(1))
    log_ret.fillna(log_ret.mean(),inplace=True)
    num_ports = 15000

    all_weights = np.zeros((num_ports,len(stock.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):

        # Create Random Weights
        weights = np.array(np.random.random(4))

        # Rebalance Weights
        weights = weights / np.sum(weights)
        
        # Save Weights
        all_weights[ind,:] = weights

        # Expected Return
        ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    
    maxSharpe = sharpe_arr.max()
    maxSharpeIndex = sharpe_arr.argmax()
    
    return {'allocation':list(all_weights[maxSharpeIndex,:]),
            'max sharpe ratio':maxSharpe}

stocks = ['EICHERMOT.NS','TCS.NS','AXISBANK.NS','HINDUNILVR.NS']
print(optimize(stocks))