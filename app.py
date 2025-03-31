#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stock Market Portfolio Optimizer Web Application
Provides a web interface for the portfolio optimization tool
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from portfolio_optimizer.data_fetcher import fetch_stock_data
from portfolio_optimizer.optimizer import optimize_portfolio, get_efficient_frontier
from portfolio_optimizer.visualizer import plot_efficient_frontier, plot_portfolio_composition, plot_historical_performance

app = Flask(__name__)

@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Handle portfolio optimization request
    """
    # Get form data
    data = request.get_json()
    tickers = [ticker.strip() for ticker in data.get('tickers', '').split(',')]
    start_date = data.get('start_date', '2018-01-01')
    end_date = data.get('end_date', None)
    risk_free_rate = float(data.get('risk_free_rate', 0.02))
    optimization_goal = data.get('optimization_goal', 'max_sharpe')
    
    # Validate inputs
    if not tickers or '' in tickers:
        return jsonify({'error': 'Please enter at least one ticker symbol'}), 400
    
    # Fetch stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    # Check if data was successfully fetched
    if stock_data is None or stock_data.empty:
        return jsonify({'error': 'Failed to fetch stock data. Please check your tickers and try again.'}), 400
    
    # Optimize portfolio
    try:
        optimized_weights, expected_return, expected_risk, sharpe_ratio = optimize_portfolio(
            stock_data, 
            risk_free_rate=risk_free_rate,
            optimization_goal=optimization_goal
        )
    except Exception as e:
        return jsonify({'error': f'Error during portfolio optimization: {str(e)}'}), 400
    
    # Prepare results
    weights_data = []
    for ticker, weight in zip(tickers, optimized_weights):
        weights_data.append({
            'ticker': ticker,
            'weight': round(weight * 100, 2)  # Convert to percentage with 2 decimal places
        })
    
    # Generate visualizations
    plots = generate_plots(stock_data, optimized_weights, tickers)
    
    # Return results
    return jsonify({
        'weights': weights_data,
        'metrics': {
            'expected_return': round(expected_return * 100, 2),  # Convert to percentage
            'expected_risk': round(expected_risk * 100, 2),  # Convert to percentage
            'sharpe_ratio': round(sharpe_ratio, 2)
        },
        'plots': plots
    })

def generate_plots(returns, weights, tickers):
    """
    Generate plots for the optimized portfolio
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame of historical returns for each asset
    weights : numpy.ndarray
        Array of optimized weights for each asset
    tickers : list
        List of ticker symbols
        
    Returns:
    --------
    dict
        Dictionary containing base64-encoded plot images
    """
    plots = {}
    
    # Plot 1: Efficient Frontier
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_efficient_frontier(returns, ax1)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    plots['efficient_frontier'] = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)
    
    # Plot 2: Portfolio Composition
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_portfolio_composition(weights, tickers, ax2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    plots['portfolio_composition'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)
    
    # Plot 3: Historical Performance
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_historical_performance(returns, weights, ax3)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
    buf3.seek(0)
    plots['historical_performance'] = base64.b64encode(buf3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
    return plots

if __name__ == '__main__':
    app.run(debug=True)