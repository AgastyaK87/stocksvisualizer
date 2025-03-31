#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stock Market Portfolio Optimizer
Main entry point for the application
"""

import argparse
from portfolio_optimizer.data_fetcher import fetch_stock_data
from portfolio_optimizer.optimizer import optimize_portfolio
from portfolio_optimizer.visualizer import visualize_portfolio

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Stock Market Portfolio Optimizer')
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'DIS', 'NFLX', 'INTC', 'KO', 'PEP', 'WMT', 'HD', 'MRK', 'VZ'],
                        help='List of stock tickers to include in the portfolio')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD), defaults to today')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Risk-free rate for portfolio optimization')
    parser.add_argument('--optimization-goal', type=str, choices=['max_sharpe', 'min_volatility'], 
                        default='max_sharpe', help='Optimization goal')
    
    return parser.parse_args()

def main():
    """
    Main function to run the portfolio optimizer
    """
    args = parse_arguments()
    
    print(f"Fetching data for {args.tickers}...")
    stock_data = fetch_stock_data(args.tickers, args.start_date, args.end_date)
    
    # Check if data was successfully fetched
    if stock_data is None or stock_data.empty:
        print("Error: Failed to fetch stock data. Please check your tickers and try again.")
        return
        
    print(f"Successfully fetched data for {len(stock_data.columns)} stocks over {len(stock_data)} days.")
    print(f"Optimizing portfolio with goal: {args.optimization_goal}...")
    
    try:
        optimized_weights, expected_return, expected_risk, sharpe_ratio = optimize_portfolio(
            stock_data, 
            risk_free_rate=args.risk_free_rate,
            optimization_goal=args.optimization_goal
        )
    except Exception as e:
        print(f"Error during portfolio optimization: {str(e)}")
        return
    
    # Print results
    print("\nOptimized Portfolio Weights:")
    for ticker, weight in zip(args.tickers, optimized_weights):
        print(f"{ticker}: {weight:.4f}")
    
    print(f"\nExpected Annual Return: {expected_return:.4f}")
    print(f"Expected Annual Risk: {expected_risk:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Visualize results
    visualize_portfolio(stock_data, optimized_weights, args.tickers)

if __name__ == "__main__":
    main()