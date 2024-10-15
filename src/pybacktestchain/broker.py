import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from pybacktestchain.utils import generate_random_name


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime
#---------------------------------------------------------
# Classes
#---------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: int
    entry_price: float

@dataclass
class Broker:
    cash: float 
    positions: dict = None # the current positions 
    transaction_log: pd.DataFrame = None #the history of all positions taken
    entry_prices: dict = None

    def __post_init__(self): #because we use dataclass we use post init and not init directly - when I define a class then i need to define a constructor 
        # Initialize positions as a dictionary of Position objects
        if self.positions is None:
            self.positions = {}
        # Initialize the transaction log as an empty DataFrame if none is provided
        if self.transaction_log is None:
            self.transaction_log = pd.DataFrame(columns=['Date', 'Action', 'Ticker', 'Quantity', 'Price', 'Cash'])
    
        # Initialize the entry prices as a dictionary
        if self.entry_prices is None:
            self.entry_prices = {}

    def get_cash_balance(self):
        """Returns the current cash balance."""
        return self.cash

    def buy(self, ticker: str, quantity: int, price: float, date: datetime):
        """Executes a buy order for the specified ticker."""
        total_cost = price * quantity ## we can add fees, transaction fees if you want to make it more complexe
        if self.cash >= total_cost: # do I have money for that 
            self.cash -= total_cost # need to be substracted 
            if ticker in self.positions:
                # Update existing position
                position = self.positions[ticker]
                new_quantity = position.quantity + quantity
                new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                position.quantity = new_quantity
                position.entry_price = new_entry_price
            else:
                # Create new position
                self.positions[ticker] = Position(ticker, quantity, price)
            # Log the transaction
            self.log_transaction(date, 'BUY', ticker, quantity, price)

            # store the entry prices 
            self.entry_prices[ticker] = price
        else:
            logging.warning(f"Not enough cash to buy {quantity} shares of {ticker} at {price}. Available cash: {self.cash}")
    
    def sell(self, ticker: str, quantity: int, price: float, date: datetime): # we do not have still the short sell, here it's only selling what I already have
        """Executes a sell order for the specified ticker."""
        if ticker in self.positions and self.positions[ticker].quantity >= quantity:
            position = self.positions[ticker]
            position.quantity -= quantity
            self.cash += price * quantity
            # If position size becomes zero, remove it
            if position.quantity == 0:
                del self.positions[ticker]
                del self.entry_prices[ticker]
            # Log the transaction
            self.log_transaction(date, 'SELL', ticker, quantity, price)
        else:
            logging.warning(f"Not enough shares to sell {quantity} shares of {ticker}. Position size: {self.positions.get(ticker, 0)}")
    
    def log_transaction(self, date, action, ticker, quantity, price):
        """Logs the transaction."""
        transaction = pd.DataFrame([{
            'Date': date,
            'Action': action,
            'Ticker': ticker,
            'Quantity': quantity,
            'Price': price,
            'Cash': self.cash
        }])
        # Concatenate the new transaction to the existing log
        self.transaction_log = pd.concat([self.transaction_log, transaction], ignore_index=True)

    def get_portfolio_value(self, market_prices: dict):
        """Calculates the total portfolio value based on the current market prices."""
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            portfolio_value += position.quantity * market_prices[ticker]
        return portfolio_value
    
    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
        """Executes the trades for the portfolio based on the generated weights."""
        ###### Pull request to have one instead of two loops 
        # First, handle all the sell orders to free up cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            # Calculate the desired quantity based on portfolio weight
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            # First, execute the sell trades (if quantity_to_trade is negative)
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
        
        # Then, handle all the buy orders, checking if there's enough cash
        #for ticker, weight in portfolio.items():
        #    price = prices.get(ticker)
        #    if price is None:
        #        logging.warning(f"Price for {ticker} not available on {date}")
        #        continue
            
            # Calculate the desired quantity based on portfolio weight
        #    total_value = self.get_portfolio_value(prices)
        #    target_value = total_value * weight
        #    current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
        #    diff_value = target_value - current_value
        #    quantity_to_trade = int(diff_value / price)
            
            # Now, execute the buy trades (if quantity_to_trade is positive) and check if there's enough cash
            elif quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date)
                else:
                    logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                    # then buy as many shares as possible with the available cash
                    logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(ticker, quantity_to_trade, price, date)

    def get_transaction_log(self):
        """Returns the transaction log."""
        return self.transaction_log

@dataclass
class RebalanceFlag: #### to make returning boolean true or false - function of today
    def time_to_rebalance(self, t: datetime):
        pass 

# Implementation of e.g. rebalancing at the end of each month
@dataclass
class EndOfMonth(RebalanceFlag):
    def time_to_rebalance(self, t: datetime): ##Change the function
        # Convert to pandas Timestamp for convenience
        pd_date = pd.Timestamp(t)
        # Get the last business day of the month
        last_business_day = pd_date + pd.offsets.BMonthEnd(0)
        # Check if the given date matches the last business day
        return pd_date == last_business_day

@dataclass
class RiskModel:
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict):
        pass

@dataclass
class StopLoss(RiskModel):
    threshold: float = 0.1
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: Broker):
        
        for ticker, position in list(broker.positions.items()):
            entry_price = broker.entry_prices[ticker]
            current_price = prices.get(ticker)
            if current_price is None:
                logging.warning(f"Price for {ticker} not available on {t}")
                continue
            # Calculate the loss percentage
            loss = (current_price - entry_price) / entry_price
            if loss < -self.threshold:
                logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all shares.")
                broker.sell(ticker, position.quantity, current_price, t)
@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag : type = EndOfMonth
    risk_model : type = StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio

    broker = Broker(cash=initial_cash)

    def __post_init__(self):
        self.backtest_name = generate_random_name()

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe")
        self.risk_model = self.risk_model(threshold=0.1)
        # self.initial_date to yyyy-mm-dd format
        init_ = self.initial_date.strftime('%Y-%m-%d')
        # self.final_date to yyyy-mm-dd format
        final_ = self.final_date.strftime('%Y-%m-%d')
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s = self.s, 
                                    data_module = data_module,
                                    time_column=self.time_column,
                                    company_column=self.company_column,
                                    adj_close_column=self.adj_close_column)
        
        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
           
            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()
        # save to csv, use the backtest name 
        df.to_csv(f"backtests/{self.backtest_name}.csv")
    