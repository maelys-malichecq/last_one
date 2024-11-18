import pytest
import pandas as pd
from datetime import datetime
from pybacktestchain.broker import Broker, Position

# Test cases
def test_broker_initialization():
    broker = Broker(cash=1000)
    assert broker.cash == 1000
    assert isinstance(broker.positions, dict)
    assert broker.transaction_log.empty


def test_broker_buy():
    broker = Broker(cash=1000)
    broker.buy(ticker="AAPL", quantity=5, price=100, date=datetime(2024, 1, 1))
    
    assert broker.cash == 500
    assert "AAPL" in broker.positions
    assert broker.positions["AAPL"].quantity == 5
    assert broker.positions["AAPL"].entry_price == 100
    assert not broker.transaction_log.empty


def test_broker_sell():
    broker = Broker(cash=1000)
    broker.buy(ticker="AAPL", quantity=5, price=100, date=datetime(2024, 1, 1))
    broker.sell(ticker="AAPL", quantity=3, price=110, date=datetime(2024, 1, 2))
    
    assert broker.cash == 830
    assert broker.positions["AAPL"].quantity == 2
    assert broker.positions["AAPL"].entry_price == 100
    assert len(broker.transaction_log) == 2  # One buy and one sell


def test_get_portfolio_value():
    broker = Broker(cash=500)
    broker.buy(ticker="AAPL", quantity=5, price=100, date=datetime(2024, 1, 1))
    market_prices = {"AAPL": 110}
    portfolio_value = broker.get_portfolio_value(market_prices)
    
    assert portfolio_value == 550  # 5 * 110
