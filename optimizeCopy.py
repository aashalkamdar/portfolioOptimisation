# Imports
from stockapi import getStockData
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
def optimize(stock):
    # stocks is an array of ticker symbols
    stock_dfs = {}
    for ticker in stock:
        print('Getting ',ticker,' data')
        stock_dfs[ticker] = getStockData(ticker)
    print('Got the stock data')
    stocks = pd.concat(list(stock_dfs.values()),axis=1)
    stocks.columns = list(stock_dfs.keys())
    
    # # simulting thousands of possible allocations

    log_ret = np.log(stocks/stocks.shift(1))

    num_ports = 15000

    all_weights = np.zeros((num_ports,len(stocks.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    print('starting some big ass for loop')
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

    print('for loop done ')
    max_sr_ret = ret_arr[7601]
    max_sr_vol = vol_arr[7601]


    plt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')


    plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')
    print('saving figure')
    plt.savefig('viz/efh1.png')

    # def get_ret_vol_sr(weights):
    
    #     #Takes in weights, returns array or return,volatility, sharpe ratio
        
    #     weights = np.array(weights)
    #     ret = np.sum(log_ret.mean() * weights) * 252
    #     vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    #     sr = ret/vol
    #     return np.array([ret,vol,sr])


    # def neg_sharpe(weights):
    #     return  get_ret_vol_sr(weights)[2] * -1

    # # Contraints
    # def check_sum(weights):
    #     '''
    #     Returns 0 if sum of weights is 1.0
    #     '''
    #     return np.sum(weights) - 1

    # cons = ({'type':'eq','fun': check_sum})



    # bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
    # init_guess = [0.25,0.25,0.25,0.25]

    # opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)


    # frontier_y = np.linspace(0,0.3,100)


    # def minimize_volatility(weights):
    #     return  get_ret_vol_sr(weights)[1] 


    # frontier_volatility = []

    # for possible_return in frontier_y:
    #     # function for return
    #     cons = ({'type':'eq','fun': check_sum},
    #             {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
        
    #     result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
        
    #     frontier_volatility.append(result['fun'])


    # plt.figure(figsize=(12,8))
    # plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    # plt.colorbar(label='Sharpe Ratio')
    # plt.xlabel('Volatility')
    # plt.ylabel('Return')

    # # Add frontier line
    # plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
    # plt.savefig('./viz/efh.png')


stocks = ['AAPL','MSFT','SNAP','AMZN']
optimize(stocks)