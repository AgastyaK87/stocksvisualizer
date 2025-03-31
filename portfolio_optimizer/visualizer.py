#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer Module
Handles visualization of portfolio optimization results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from portfolio_optimizer.optimizer import get_efficient_frontier

# Set style
sns.set_theme(style="whitegrid")

def visualize_portfolio(returns, weights, tickers, save_path=None):
    """
    Visualize portfolio optimization results
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    weights : numpy.ndarray
        Array of optimized weights for each asset
    tickers : list
        List of ticker symbols
    save_path : str, optional
        Path to save the visualization, if None, display the plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Portfolio Optimization Results', fontsize=16)
    
    # Plot 1: Efficient Frontier
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    plot_efficient_frontier(returns, ax1)
    
    # Plot 2: Portfolio Composition
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    plot_portfolio_composition(weights, tickers, ax2)
    
    # Plot 3: Historical Performance
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plot_historical_performance(returns, weights, ax3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_efficient_frontier(returns, ax=None):
    """
    Plot the efficient frontier
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, create a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate efficient frontier
    returns_array, volatility_array, weights_array = get_efficient_frontier(returns, num_portfolios=1000)
    
    # Plot random portfolios
    sc = ax.scatter(volatility_array, returns_array, 
               c=returns_array/volatility_array, s=10, alpha=0.3,
               cmap='viridis', label='Portfolios')
    
    # Find and plot the minimum volatility portfolio
    min_vol_idx = np.argmin(volatility_array)
    ax.scatter(volatility_array[min_vol_idx], returns_array[min_vol_idx],
              c='red', s=50, edgecolors='black', label='Minimum Volatility')
    
    # Find and plot the maximum Sharpe ratio portfolio
    sharpe_array = returns_array / volatility_array
    max_sharpe_idx = np.argmax(sharpe_array)
    ax.scatter(volatility_array[max_sharpe_idx], returns_array[max_sharpe_idx],
              c='green', s=50, edgecolors='black', label='Maximum Sharpe Ratio')
    
    # Plot individual assets
    for i, ticker in enumerate(returns.columns):
        # Create a portfolio with 100% in this asset
        single_asset_return = returns.iloc[:, i].mean() * 252
        single_asset_volatility = returns.iloc[:, i].std() * np.sqrt(252)
        ax.scatter(single_asset_volatility, single_asset_return,
                  c='blue', s=50, alpha=0.7, label=ticker)
        ax.annotate(ticker, (single_asset_volatility, single_asset_return),
                   xytext=(4, 4), textcoords='offset points')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    # Set labels and title
    ax.set_xlabel('Expected Volatility (Annualized)')
    ax.set_ylabel('Expected Return (Annualized)')
    ax.set_title('Efficient Frontier')
    
    # Add legend (only for min vol and max sharpe)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = ['Portfolios', 'Minimum Volatility', 'Maximum Sharpe Ratio'] + list(returns.columns)
    unique_handles = [handles[labels.index(label)] for label in unique_labels if label in labels]
    ax.legend(unique_handles, unique_labels, loc='best', frameon=True)
    
    return ax

def plot_portfolio_composition(weights, tickers, ax=None):
    """
    Plot portfolio composition as a pie chart
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Array of optimized weights for each asset
    tickers : list
        List of ticker symbols
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, create a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter out assets with very small weights
    threshold = 0.01  # 1%
    significant_indices = weights > threshold
    significant_weights = weights[significant_indices]
    significant_tickers = [tickers[i] for i, is_significant in enumerate(significant_indices) if is_significant]
    
    # If there are insignificant weights, add an 'Other' category
    if sum(significant_indices) < len(weights):
        other_weight = sum(weights[~significant_indices])
        if other_weight > 0:
            significant_weights = np.append(significant_weights, other_weight)
            significant_tickers.append('Other')
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        significant_weights, 
        labels=significant_tickers,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
    
    ax.set_title('Optimized Portfolio Composition')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    return ax

def plot_historical_performance(returns, weights, ax=None):
    """
    Plot historical performance of the optimized portfolio
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    weights : numpy.ndarray
        Array of optimized weights for each asset
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, create a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate portfolio returns over time
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # Plot cumulative returns
    ax.plot(cumulative_returns.index, cumulative_returns * 100, linewidth=2, label='Optimized Portfolio')
    
    # Plot individual assets
    for i, ticker in enumerate(returns.columns):
        asset_cumulative_returns = (1 + returns.iloc[:, i]).cumprod() - 1
        ax.plot(asset_cumulative_returns.index, asset_cumulative_returns * 100, 
                alpha=0.5, linewidth=1, label=ticker)
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Historical Performance Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    return ax

def plot_correlation_matrix(returns, ax=None):
    """
    Plot correlation matrix of asset returns
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, create a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax)
    
    ax.set_title('Correlation Matrix of Asset Returns')
    
    return ax