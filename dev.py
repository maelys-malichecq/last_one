# %%
from src.pybacktestchain.data_module import get_stocks_data, UNIVERSE_SEC, DataModule, Information, FirstTwoMoments
from datetime import timedelta

# pick 10 random stocks
import random
random.seed(42)
stocks = random.sample(UNIVERSE_SEC, 10)

df = get_stocks_data(stocks, '2020-01-01', '2022-01-01')

# Initialize the DataModule
data_module = DataModule(df)

# Create the FirstTwoMoments object
info = FirstTwoMoments(s = timedelta(days=360), 
                       data_module = data_module,
                       time_column='Date',
                       company_column='ticker',
                       adj_close_column='Adj Close')


# %% Normal portfolio optimization like we did in IMF
t = df['Date'].max()
information_set = info.compute_information(t)

portfolio = info.compute_portfolio(t, information_set)

print("Success")
print(t)
print(information_set)
print(portfolio) #observe the weights


# %% Minimum Variance portfolio; it works
t = df['Date'].max()
information_set = info.compute_information(t)

# Run the basic minimum variance portfolio optimization
portfolio_min_variance = info.compute_min_variance_portfolio(t, information_set)

# Print the result of the optimization
print("Success")
print(f"Date: {t}")
print("Portfolio (minimum variance):", portfolio_min_variance)

# %% Target return portfolio; not working ATM, the optimzation doesnt converge
#even with a tolerance and I don't really understand why? 

target_return = 0.05  # target return of 3%
tolerance = 0.01  # Allow for a small deviation from the target return as it's too restrictive otherwise

# Compute the information set for the given date
information_set = info.compute_information(t)

# Run the portfolio optimization with the target return
portfolio_tgt_return = info.compute_portfolio_tgt_return(t, information_set, target_return, tolerance)

# Output the results
print("Success")
print(f"Date: {t}")
print(f"Target Return: {target_return}")
print(f"Tolerance: {tolerance}")
print("Portfolio (target return):", portfolio_tgt_return)

# %%
