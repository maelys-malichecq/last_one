
import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np

import sys
import os

# Add the directory to the system path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys())

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

# function that retrieves historical data on prices for a given stock
def get_stock_data(ticker, start_date, end_date):
    """get_stock_data retrieves historical data on prices for a given stock

    Args:
        ticker (str): The stock ticker
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data

    Example:
        df = get_stock_data('AAPL', '2000-01-01', '2020-12-31')
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
    # as dataframe 
    df = pd.DataFrame(data)
    df['ticker'] = ticker
    df.reset_index(inplace=True)
    return df

def get_stocks_data(tickers, start_date, end_date):
    """get_stocks_data retrieves historical data on prices for a list of stocks

    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data

    Example:
        df = get_stocks_data(['AAPL', 'MSFT'], '2000-01-01', '2020-12-31')
    """
    # get the data for each stock
    # try/except to avoid errors when a stock is not found
    dfs = []
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            # append if not empty
            if not df.empty:
                dfs.append(df)
        except:
            logging.warning(f"Stock {ticker} not found")
    # concatenate all dataframes
    data = pd.concat(dfs)
    return data

#---------------------------------------------------------
# Classes 
#---------------------------------------------------------

# Class that represents the data used in the backtest. 
@dataclass
class DataModule:
    data: pd.DataFrame

# Interface for the information set 
@dataclass
class Information:
    s: timedelta # Time step (rolling window)
    data_module: DataModule # Data module
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t : datetime):
         # Get the data module 
        data = self.data_module.data
        # Get the time step 
        s = self.s
        # Get the data only between t-s and t
        data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
        return data

    def compute_information(self, t : datetime):  
        pass

    def compute_portfolio(self, t : datetime,  information_set : dict):
        pass
       
        
@dataclass
class FirstTwoMoments(Information):

    def compute_portfolio(self, t:datetime, information_set,risk_free_rate=0.01):
        mu = information_set['expected_return']
        Sigma = information_set['covariance_matrix']
        kurtosis = information_set['kurtosis']

        #gamma = 1  # risk aversion parameter
        n = len(mu)
        # objective function : Minimize Kurtosis 
        obj = lambda x: x.dot(kurtosis)
        #obj = lambda x: -x.dot(mu) + gamma / 2 * x.dot(Sigma).dot(x)
        # constraints
        cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}),  # sum of weights = 1
        #{'type': 'ineq', 'fun': lambda x: x.dot(mu)})     # expected return > 0
        
        # bounds, allow short selling, +- inf
        bounds = [(0, None)] * n
        # initial guess, equal weights
        x0 = np.ones(n) / n
        # minimize
        res = minimize(lambda x: -(x.dot(mu) - risk_free_rate) / np.sqrt(x.dot(Sigma).dot(x)),x0,constraints=cons,bounds=bounds)

        # prepare dictionary
        portfolio = {k: None for k in information_set['companies']}

        # if converged update
        if res.success:
            for i, company in enumerate(information_set['companies']):
                portfolio[company] = res.x[i]

        return portfolio

    def compute_information(self, t: datetime):
        # Get the data module
        data = self.slice_data(t)
        # the information set will be a dictionary with the data
        information_set = {}

        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # expected return per company
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        
        # expected return by company
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix
        data_pivot = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data_pivot = data_pivot.dropna(axis=0)
        covariance_matrix = data_pivot.cov().to_numpy()
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        # Skewness and Kurtosis
        skewness = data.groupby(self.company_column)['return'].skew().to_numpy()
        kurtosis = data.groupby(self.company_column)['return'].apply(pd.Series.kurt).to_numpy()
        
        information_set['skewness'] = skewness
        information_set['kurtosis'] = kurtosis
        
        return information_set


