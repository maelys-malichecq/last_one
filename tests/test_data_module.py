import pytest
from datetime import datetime, timedelta
from pybacktestchain.data_module import get_stock_data, get_stocks_data, DataModule, Information
import pandas as pd

# Mock data for testing
MOCK_DATA = pd.DataFrame({
    "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
    "ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
    "Close": [150, 152, 148, 151, 153]
})

def test_get_stock_data():
    """Test that stock data is retrieved as a DataFrame."""
    data = get_stock_data("AAPL", "2024-01-01", "2024-01-05")
    assert isinstance(data, pd.DataFrame)
    assert "ticker" in data.columns
    assert not data.empty


def test_get_stocks_data():
    """Test retrieving data for multiple stocks."""
    data = get_stocks_data(["AAPL", "MSFT"], "2024-01-01", "2024-01-05")
    assert isinstance(data, pd.DataFrame)
    assert "ticker" in data.columns
    assert not data.empty
    assert data["ticker"].nunique() == 2


def test_data_module():
    """Test DataModule initialization and slicing."""
    module = DataModule(data=MOCK_DATA)
    assert isinstance(module.data, pd.DataFrame)
    
    # Test slicing method
    t = datetime(2024, 1, 5)
    information = Information(s=timedelta(days=3), data_module=module)
    sliced_data = information.slice_data(t)
    assert len(sliced_data) == 3  # Last three days in data


def test_information_get_prices():
    """Test getting last available prices from data slice."""
    module = DataModule(data=MOCK_DATA)
    info = Information(s=timedelta(days=3), data_module=module)
    prices = info.get_prices(datetime(2024, 1, 6))
    assert prices == {"AAPL": 153}  # Should return last price for AAPL
