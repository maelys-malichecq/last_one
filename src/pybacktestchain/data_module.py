
import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------
# for the moment, we assume that the universe of companies is constant 

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys()) #we ask for the keys of the dictionnary SotckMapper

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
# Classes r
#---------------------------------------------------------

# Class that represents the data used in the backtest. 
@dataclass
class DataModule:
    data: pd.DataFrame

# Interface for the information set  
    ##Jean##  (an interface is a class that enable to create other classes easily.
    ##Jean##  it's a class that can be called in another class)
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
    ## the 2 functions below will be modified depending on the backtest we want to do (we want to write them)
    def compute_information(self, t : datetime):  
        pass

    def compute_portfolio(self, t : datetime,  information_set : dict):
        pass
       
        
@dataclass
class FirstTwoMoments(Information): #inherite the information class computed just above

    def compute_portfolio(self, t:datetime, information_set):
        mu = information_set['expected_return']
        Sigma = information_set['covariance_matrix']

        gamma = 1 # risk aversion parameter
        n = len(mu)
        # objective function
        obj = lambda x: -x.dot(mu) + gamma/2 * x.dot(Sigma).dot(x) # x is the vector of weights : see the course of IMF for the matrices (we have a (- because we minimize))
        # constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # we have an equality constraint, here sum(x) = 1 <=> sum(x) - 1 = 0, the sum of weights needs to equal 1
        # bounds, allow short selling, +- inf 
        bounds = [(None, None)] * n    # no bound constraint here for all element
        # initial guess, equal weights
        x0 = np.ones(n) / n     #at first, we have an equally weighted portfolio (make sure that the initial portfolio is feasible with our constraints)
        # minimize
        res = minimize(obj, x0, constraints=cons, bounds=bounds)

        # prepare dictionary 
        portfolio = {k: None for k in information_set['companies']}

        # if converged update
        if res.success:
            for i, company in enumerate(information_set['companies']):
                portfolio[company] = res.x[i]
        
        return portfolio

    def compute_information(self, t : datetime):
        # Get the data module 
        data = self.slice_data(t)
        # the information set will be a dictionary with the data
        information_set = {}

        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # expected return per company
        data['return'] =  data.groupby(self.company_column)[self.adj_close_column].pct_change()
        data = data.dropna(subset='return') # to erase the lines with no returns
        
        # expected return by company 
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        ## covariance matrix :

        # 1. pivot the data
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        # drop missing values
        data = data.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data.cov()
        # convert to numpy matrix 
        covariance_matrix = covariance_matrix.to_numpy()
        # add to the information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        return information_set
    
    #######################################################################################################################
    # let's create a portfolio based on the Risk Parity, where each asset in the Portfolio contributes the same amount of risk.

    def compute_Risky_Parity_portfolio(self, t:datetime, information_set):
        mu = information_set['expected_return']
        Sigma = information_set['covariance_matrix']

        gamma = 1 # risk aversion parameter
        n = len(mu)
        # objective function
        def obj(x):
            # Portfolio volatility
            pf_vol = np.sqrt(x.dot(Sigma).dot(x))
            # Risk contribution of each asset
            risk_contrib = x.dot(Sigma).dot(x)
            # Target risk contribution : we want equal risk contribution for each asset
            target_risk_contrib = pf_vol / n
            # Objective : minimize the sqrt deviation form equal risk contribution
            return np.sum((risk_contrib - target_risk_contrib) ** 2)

        # constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # we have an equality constraint, here sum(x) = 1 <=> sum(x) - 1 = 0, the sum of weights needs to equal 1
        # bounds, allow short selling, +- inf 
        bounds = [(0, 0.6)] * n    # in the Risk Parity Porfolion there is no short selling, and stricter bounds to prevent extreme weights
        # initial guess, equal weights
        x0 = np.ones(n) / n     #at first, we have an equally weighted portfolio (make sure that the initial portfolio is feasible with our constraints)
        # minimize
        res = minimize(obj, x0, constraints=cons, bounds=bounds)

        # prepare dictionary 
        portfolio = {k: None for k in information_set['companies']}

        # if converged update
        if res.success:
            for i, company in enumerate(information_set['companies']):
                portfolio[company] = res.x[i]
        
        return portfolio