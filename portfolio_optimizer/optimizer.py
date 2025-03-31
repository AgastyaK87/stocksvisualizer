#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Optimizer Module
Implements portfolio optimization algorithms using Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_performance(weights, returns):
    """
    Calculate portfolio performance metrics
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Array of weights for each asset
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
        
    Returns:
    --------
    tuple
        (portfolio_return, portfolio_volatility, sharpe_ratio)
    """
    # Convert to numpy array if needed
    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
    else:
        returns_array = returns
    
    # Calculate portfolio return and volatility (annualized)
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0 for now)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calculate negative Sharpe ratio (for minimization)
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Array of weights for each asset
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    float
        Negative Sharpe ratio
    """
    portfolio_return, portfolio_volatility, _ = calculate_portfolio_performance(weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def portfolio_volatility(weights, returns):
    """
    Calculate portfolio volatility
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Array of weights for each asset
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
        
    Returns:
    --------
    float
        Portfolio volatility (annualized)
    """
    return calculate_portfolio_performance(weights, returns)[1]

def optimize_portfolio(returns, risk_free_rate=0.02, optimization_goal='max_sharpe'):
    """
    Optimize portfolio weights based on specified goal
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    risk_free_rate : float
        Risk-free rate (annualized)
    optimization_goal : str
        Optimization objective ('max_sharpe' or 'min_volatility')
        
    Returns:
    --------
    tuple
        (optimized_weights, expected_return, expected_risk, sharpe_ratio)
    """
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    if optimization_goal == 'max_sharpe':
        # Maximize Sharpe ratio
        optimization_result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    elif optimization_goal == 'min_volatility':
        # Minimize volatility
        optimization_result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        raise ValueError(f"Unknown optimization goal: {optimization_goal}")
    
    # Get optimized weights
    optimized_weights = optimization_result['x']
    
    # Calculate expected performance
    expected_return, expected_risk, sharpe = calculate_portfolio_performance(optimized_weights, returns)
    sharpe_ratio = (expected_return - risk_free_rate) / expected_risk
    
    return optimized_weights, expected_return, expected_risk, sharpe_ratio

def get_efficient_frontier(returns, num_portfolios=100):
    """
    Generate the efficient frontier
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    num_portfolios : int
        Number of portfolios to generate
        
    Returns:
    --------
    tuple
        (returns_array, volatility_array, weights_array)
    """
    num_assets = len(returns.columns)
    results_return = []
    results_volatility = []
    results_weights = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility, _ = calculate_portfolio_performance(weights, returns)
        
        # Store results
        results_return.append(portfolio_return)
        results_volatility.append(portfolio_volatility)
        results_weights.append(weights)
    
    return np.array(results_return), np.array(results_volatility), np.array(results_weights)