
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from stockapi import getStockData


# In[3]:

stocks = ['AAPL','MSFT','SNAP','AMZN']

for stock in stocks:
    print(stock)
    print(getStockData(stock))

# In[4]:


stock1 = pd.read_csv(a)
stock2 = pd.read_csv(b)
stock3 = pd.read_csv(c)
stock4 = pd.read_csv(d)


# In[9]:





# In[10]:
def preprocess(stock):
    stock.drop('Open',axis=1,inplace=True)
    stock.drop('High',axis=1,inplace=True)
    stock.drop('Low',axis=1,inplace=True)
    stock.drop('Close',axis=1,inplace=True)
    stock.drop('Volume',axis=1,inplace=True)
    stock['Date'] = pd.to_datetime(stock1['Date'])
    stock.set_index('Date',inplace=True)


for st


# In[12]:


stocks = pd.concat([stock1,stock2,stock3,stock4],axis=1)
stocks.columns = [a,b,c,d]


# In[13]:


stocks.head()


# # Monte Carlo Simultation

# In[14]:


mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret


# In[15]:


stocks.pct_change(1).corr()


# # simulting thousands of possible allocations

# In[16]:


stock_normed = stocks/stocks.iloc[0]
stock_normed.plot()


# In[17]:


stock_daily_ret = stocks.pct_change(1)
stock_daily_ret.head()


# # log returns

# In[18]:


log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()


# In[19]:


log_ret.hist(bins=100);
plt.tight_layout()


# In[20]:


log_ret.describe()


# In[21]:


log_ret.mean() * 252


# In[22]:


log_ret.cov()


# In[23]:


log_ret.cov()*252 


# In[25]:


num_ports = 15000

all_weights = np.zeros((num_ports,len(stocks.columns)))
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


# In[26]:


sharpe_arr.max()


# In[27]:


sharpe_arr.argmax()


# In[28]:


all_weights[7601,:]


# In[29]:


#the following are the allocations according to the stock


# In[30]:


max_sr_ret = ret_arr[7601]
max_sr_vol = vol_arr[7601]


# # plotting

# In[31]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')


# # more optimised way rather than running 15000 simulation

# In[43]:


def get_ret_vol_sr(weights):
   
    #Takes in weights, returns array or return,volatility, sharpe ratio
    
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])


# In[44]:


from scipy.optimize import minimize


# In[45]:


def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1


# In[46]:


# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1


# In[47]:


cons = ({'type':'eq','fun': check_sum})


# In[48]:


bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
init_guess = [0.25,0.25,0.25,0.25]


# In[49]:


opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)


# In[50]:


opt_results


# In[51]:


opt_results.x


# In[52]:


get_ret_vol_sr(opt_results.x)


# In[57]:


# efficient frontier hypothesis ( not quite imp)
# gives the best return for a particular volatility


# In[53]:


frontier_y = np.linspace(0,0.3,100)


# In[54]:


def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] 


# In[55]:


frontier_volatility = []

for possible_return in frontier_y:
    # function for return
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    
    frontier_volatility.append(result['fun'])


# In[56]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')



# Add frontier line
plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)

