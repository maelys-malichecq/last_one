
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
#%% 

#create a class that give the expected return of a portfolio with a high volatility environment

class hedge_fund_vol(Information):

    def compute_portfolio(self, t:datetime, information_set):
        mu = information_set['expected_return']
        Sigma = information_set['covariance_matrix']

        gamma = 0.1 # risk aversion parameter
        n = len(mu)
        # objective function
        obj = lambda x: -x.dot(mu) + gamma/2 * x.dot(Sigma).dot(x) #Maximising the return while minimizing vol
        '''We will use the CVaR (or Expected Shortfall) which is the weighted average of observations whose losses are greated than
        the VaR'''

        # Tail risk constraint: Compute CVaR (Expected Shortfall) using mu
        '''Assumption that returns are normally distributed to be able to compute Var and CVar, otherwise would have to simulate
        returns with Monte Carlo simulation'''

        alpha = 0.95  # Confidence level for VaR and CVaR (95% confidence)
        z_alpha = -1.96  # Critical value for 95% confidence level (VaR), we take the negative value as we want to find what is below the mean

        def VaR_calculation(weights):
            #Calculate the portfolio variance (volatility squared)
            ptf_std= np.sqrt(weights.dot(Sigma).dot(weights))
            #Calculating the portfolio's expected return
            ptf_average = np.dot(mu, weights)
            #Calculating Value-at-Risk (VaR) using mu and the portfolio variance
            VaR = ptf_average + z_alpha * ptf_std  # VaR at 95% confidence
            return VaR, ptf_std

        def CVaR_calculation(weights):
            #Calculate Conditional Value-at-Risk (CVaR or Expected Shortfall) using mu
            VaR = VaR_calculation(weights)
            #Since returns are assumed to be normally distributed, CVaR can be calculated
            ptf_std = VaR_calculation(weights)[1]
            #computing the expected shortfall ES
            ES = VaR - (ptf_std(weights) * (np.exp(-z_alpha ** 2 / 2) / (alpha * np.sqrt(2 * np.pi))))
            return ES

        # Constraints:
        # 1. Sum of all weight should be equal to 1
        # 2. Portfolio's expected return should be greater than 10%
        # 3. Portfolio Conditional Value-at-Risk should be less than 1% of the portfolio value
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},   # Sum of weights = 1
                {'type': 'ineq', 'fun': lambda x: x.dot(mu)-0.1},     # Expected return > 10%
                {'type': 'ineq', 'fun': lambda x: 0.01 + CVaR_calculation(x)}]  # CVaR < 1% of portfolio value
    
        # bounds, allow short selling, +- inf 
        bounds = [(None, None)] * n

        # initial guess, equally weighted portfolio
        x0 = np.ones(n) / n #have to make sure that the initial point is feasible (inside the constraint)
        
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
        data['return'] =  data.groupby(self.company_column)[self.adj_close_column].pct_change() #.mean()
        
        # expected return by company 
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

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
    



