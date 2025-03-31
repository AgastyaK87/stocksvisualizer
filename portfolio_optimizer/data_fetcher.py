#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Fetcher Module
Handles retrieving stock data from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

def fetch_stock_data(tickers, start_date, end_date=None):
    """
    Fetch historical stock data from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD', defaults to today
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing adjusted close prices for the specified stocks
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    try:
        # Download data for all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Check if data is empty or None
        if data is None or data.empty:
            print(f"Error: No data returned for tickers {tickers}")
            return None
            
        # Print data structure for debugging
        print(f"Data type: {type(data)}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns}")
        
        # Handle different data structures based on the response
        if isinstance(data, pd.DataFrame):
            # For a single ticker
            if len(tickers) == 1:
                if 'Adj Close' in data.columns:
                    adj_close = data['Adj Close'].to_frame()
                    adj_close.columns = [tickers[0]]
                elif 'Close' in data.columns:
                    print("Warning: 'Adj Close' not found, using 'Close' instead")
                    adj_close = data['Close'].to_frame()
                    adj_close.columns = [tickers[0]]
                else:
                    print(f"Error: Price data not found. Available columns: {data.columns}")
                    return None
            # For multiple tickers
            else:
                if 'Adj Close' in data.columns.levels[0]:
                    adj_close = data['Adj Close']
                elif 'Close' in data.columns.levels[0]:
                    print("Warning: 'Adj Close' not found, using 'Close' instead")
                    adj_close = data['Close']
                else:
                    print(f"Error: Price data not found. Available column levels: {data.columns.levels}")
                    return None
        
        # Check for missing data
        missing_data = adj_close.isna().sum()
        if missing_data.any():
            print("Warning: Missing data detected for some tickers:")
            for ticker, count in missing_data[missing_data > 0].items():
                print(f"  {ticker}: {count} missing values")
            
            # Forward fill missing values
            adj_close = adj_close.fillna(method='ffill')
            # If there are still NaN values at the beginning, backward fill
            adj_close = adj_close.fillna(method='bfill')
        
        # Calculate daily returns
        returns = adj_close.pct_change().dropna()
        
        return returns
    
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def get_market_index_data(index_ticker='SPY', start_date=None, end_date=None):
    """
    Fetch market index data for comparison
    
    Parameters:
    -----------
    index_ticker : str
        Ticker symbol for market index (default: 'SPY' for S&P 500)
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD', defaults to today
        
    Returns:
    --------
    pandas.Series
        Series containing daily returns for the market index
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    try:
        index_data = yf.download(index_ticker, start=start_date, end=end_date, progress=False)
        index_returns = index_data['Adj Close'].pct_change().dropna()
        return index_returns
    
    except Exception as e:
        print(f"Error fetching market index data: {str(e)}")
        return None