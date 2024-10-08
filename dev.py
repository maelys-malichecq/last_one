#%%
from pybacktestchain.data_module import get_stocks_data, UNIVERSE_SEC, DataModule, Information, FirstTwoMoments
from datetime import timedelta

# pick 10 random stocks
import random
random.seed(42)
stocks = random.sample(UNIVERSE_SEC, 10)

df = get_stocks_data(stocks, '2000-01-01', '2020-12-31')

# Initialize the DataModule
data_module = DataModule(df)

# Create the FirstTwoMoments object
info = FirstTwoMoments(s = timedelta(days=360), 
                       data_module = data_module,
                       time_column='Date',
                       company_column='ticker',
                       adj_close_column='Adj Close')
# %%
t = df['Date'].max()
information_set = info.compute_information(t)
portfolio = info.compute_portfolio(t, information_set)

# %%
