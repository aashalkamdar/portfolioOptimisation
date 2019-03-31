import requests
import pandas as pd
import csv
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def getStockData(ticker):
    apikeyFile = open('apiKey','r')
    apikey = str(apikeyFile.readline())

    baseURL = 'https://www.alphavantage.co/query'


    PARAMS = {
        'outputsize':'full',
        'apikey'    :apikey,
        'symbol'    :ticker,
        'function'  :'TIME_SERIES_DAILY_ADJUSTED',
        'datatype'  :'csv'
    }

    data = requests.get(baseURL,PARAMS).text
    data = StringIO(data)
    stock = pd.read_csv(data)
    dropColumns = ['open','high','low','close','volume','dividend_amount','split_coefficient']
    stock.drop(dropColumns,axis=1,inplace=True)
    stock['timestamp'] = pd.to_datetime(stock['timestamp'])
    stock.set_index('timestamp',inplace=True)


    return stock

