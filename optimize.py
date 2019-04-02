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
    
    #log returns - normalising
    log_ret = np.log(stock/stock.shift(1))
    log_ret.dropna(inplace=True)
    print(log_ret.head())
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

    max_sr_ret = ret_arr[maxSharpeIndex]
    max_sr_vol = vol_arr[maxSharpeIndex]
    plt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sr_vol,max_sr_ret,c='black',s=50,edgecolors='black')
    plt.savefig('./viz/efh.png')




    # Add frontier line
    # plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
    # plt.savefig('./viz/efh1.png')
    return {'allocation':list(all_weights[maxSharpeIndex,:]),
            'max sharpe ratio':maxSharpe}
    


    # use getStockData and get all the stock data by looping through stocks

    # do the magic calculations

    # return plots and other results


stocks = ['EICHERMOT.NS','TCS.NS','AXISBANK.NS','HINDUNILVR.NS']
print(optimize(stocks))