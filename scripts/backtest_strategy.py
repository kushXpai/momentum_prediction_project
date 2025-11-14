# ==========================================
# BACKTEST MOMENTUM TRADING STRATEGY
# Purpose: Test strategy on historical data
# Date: 15th November 2025
# ==========================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("="*70)
print("BACKTESTING MOMENTUM TRADING STRATEGY")
print("="*70)

try:
    # ---- STEP 1: Load data and model ----
    print("\n[1/8] Loading data and model...")
    
    # Load all data
    labeled_df = pd.read_csv('../data/labeled_data.csv')
    labeled_df['date'] = pd.to_datetime(labeled_df['date'])
    
    # Load model and scaler
    import tensorflow as tf
    model = tf.keras.models.load_model('../models/hamnet_model.keras')
    scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
    
    print(f"✓ Loaded {len(labeled_df)} samples")
    print("✓ Loaded HAMNet model")
    print("✓ Loaded scaler")
    
    # ---- STEP 2: Prepare data for backtesting ----
    print("\n[2/8] Preparing data...")
    
    # Sort by date (chronological order)
    df = labeled_df.sort_values('date').reset_index(drop=True)
    
    # Select feature columns (same as training)
    feature_columns = ['close', 'SMA_5', 'SMA_20', 'Momentum_5', 'Momentum_10', 'Volatility_20', 'Volume_Ratio']
    
    # Get features and scale
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    
    print(f"✓ Data prepared and scaled")
    
    # ---- STEP 3: Get model predictions ----
    print("\n[3/8] Getting model predictions...")

    # Split X_scaled into the 4 model inputs
    # Feature columns order: ['close', 'SMA_5', 'SMA_20', 'Momentum_5', 'Momentum_10', 'Volatility_20', 'Volume_Ratio']

    price_input      = X_scaled[:, 0:1]        # close
    volume_input     = X_scaled[:, 1:3]        # SMA_5, SMA_20
    momentum_input   = X_scaled[:, 3:5]        # Momentum_5, Momentum_10
    volatility_input = X_scaled[:, 5:6]        # Volatility_20
    extra_volume     = X_scaled[:, 6:7]        # Volume_Ratio

    # Combine volume features into a single array with 3 features
    volume_input_combined = np.hstack([volume_input, extra_volume])

    # Make predictions
    predictions = model.predict([price_input, momentum_input, volatility_input, volume_input_combined], verbose=0)
    df['Prediction_Prob'] = predictions.flatten()

    # Convert to buy/sell signals
    confidence_threshold = 0.6
    df['Signal'] = (df['Prediction_Prob'] > confidence_threshold).astype(int)

    print(f"✓ Predictions complete")
    print(f"✓ Confidence threshold: {confidence_threshold*100:.0f}%")
    
    # ---- STEP 4: Initialize backtesting variables ----
    print("\n[4/8] Running backtest simulation...")
    
    initial_capital = 100000  # ₹100,000 starting capital
    position = 0              # Number of shares held
    cash = initial_capital    # Available cash
    
    portfolio_values = [initial_capital]
    trades = []
    daily_returns = []
    
    transaction_cost = 0.001  # 0.1% transaction fee (0.15% is realistic)
    
    # ---- STEP 5: Run backtest loop ----
    print("Simulating trades...")
    
    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]
        signal = df['Signal'].iloc[i]
        date = df['date'].iloc[i]
        
        # BUY LOGIC
        if signal == 1 and position == 0:
            # Calculate how many shares we can buy
            shares_to_buy = (cash * (1 - transaction_cost)) / current_price
            
            position = shares_to_buy
            cash = 0
            
            trades.append({
                'Date': date,
                'Type': 'BUY',
                'Price': current_price,
                'Shares': shares_to_buy,
                'Cost': current_price * shares_to_buy
            })
        
        # SELL LOGIC
        elif signal == 0 and position > 0:
            # Sell all shares at current price
            revenue = position * next_price * (1 - transaction_cost)
            
            trades.append({
                'Date': date,
                'Type': 'SELL',
                'Price': next_price,
                'Shares': position,
                'Revenue': revenue
            })
            
            cash = revenue
            position = 0
        
        # Calculate portfolio value
        if position > 0:
            # If holding shares, value = shares × current price + cash
            portfolio_value = (position * next_price) + cash
        else:
            # If not holding shares, value = cash only
            portfolio_value = cash
        
        portfolio_values.append(portfolio_value)
        daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
        daily_returns.append(daily_return)
    
    print(f"✓ Backtest complete with {len(trades)} trades")
    
    # ---- STEP 6: Calculate metrics ----
    print("\n[5/8] Calculating performance metrics...")
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)
    
    # Total return
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Annualized return (assuming ~75 trading days)
    trading_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    
    # Sharpe ratio (risk-adjusted return)
    # Assumes risk-free rate = 0
    sharpe_ratio = np.mean(daily_returns) * 252 / (np.std(daily_returns) + 1e-6)
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    max_drawdown = np.max(drawdown)
    
    # Win rate
    winning_trades = 0
    for i in range(0, len(trades)-1, 2):
        if i+1 < len(trades) and trades[i]['Type'] == 'BUY' and trades[i+1]['Type'] == 'SELL':
            buy_price = trades[i]['Price']
            sell_price = trades[i+1]['Price']
            if sell_price > buy_price:
                winning_trades += 1
    
    total_complete_trades = len(trades) // 2
    win_rate = winning_trades / total_complete_trades if total_complete_trades > 0 else 0
    
    print("✓ Metrics calculated")
    
    # ---- STEP 7: Display results ----
    print("\n" + "="*70)
    print("BACKTESTING RESULTS")
    print("="*70)
    
    print(f"\nCapital Management:")
    print(f"  Initial Capital:    ₹{initial_capital:,.0f}")
    print(f"  Final Portfolio Value: ₹{final_value:,.0f}")
    print(f"  Profit/Loss:        ₹{final_value - initial_capital:,.0f}")
    
    print(f"\nReturns:")
    print(f"  Total Return:       {total_return*100:.2f}%")
    print(f"  Annualized Return:  {annualized_return*100:.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"  Maximum Drawdown:   {max_drawdown*100:.2f}%")
    
    print(f"\nTrading Activity:")
    print(f"  Total Trades:       {len(trades)}")
    print(f"  Complete Pairs:     {total_complete_trades}")
    print(f"  Win Rate:           {win_rate*100:.2f}%")
    print(f"  Avg Daily Return:   {np.mean(daily_returns)*100:.2f}%")
    
    # ---- STEP 8: Plot backtest results ----
    print("\n[6/8] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')
    
    # Chart 1: Portfolio Value
    axes[0, 0].plot(portfolio_values, linewidth=2, color='blue')
    axes[0, 0].axhline(y=initial_capital, color='red', linestyle='--', label='Initial Capital')
    axes[0, 0].fill_between(range(len(portfolio_values)), initial_capital, portfolio_values, alpha=0.3)
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value (₹)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Daily Returns
    axes[0, 1].bar(range(len(daily_returns)), daily_returns*100, color=['green' if r > 0 else 'red' for r in daily_returns])
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_title('Daily Returns')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Drawdown
    axes[1, 0].fill_between(range(len(drawdown)), drawdown*100, alpha=0.3, color='red')
    axes[1, 0].set_title('Drawdown Over Time')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Chart 4: Cumulative Returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    axes[1, 1].plot(cumulative_returns*100, linewidth=2, color='green')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title('Cumulative Returns')
    axes[1, 1].set_ylabel('Cumulative Return (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/backtest_results.png', dpi=100)
    print("✓ Backtest chart saved")
    plt.show()
    
    # ---- STEP 9: Save backtest report ----
    print("\n[7/8] Saving backtest report...")
    
    report = f"""
BACKTEST REPORT
===============

Strategy: HAMNet Momentum Prediction
Test Period: {df['date'].min()} to {df['date'].max()}
Trading Days: {trading_days}

CAPITAL METRICS:
- Initial Capital: ₹{initial_capital:,.0f}
- Final Value: ₹{final_value:,.0f}
- Profit: ₹{final_value - initial_capital:,.0f}

PERFORMANCE METRICS:
- Total Return: {total_return*100:.2f}%
- Annualized Return: {annualized_return*100:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Maximum Drawdown: {max_drawdown*100:.2f}%

TRADING METRICS:
- Total Trades: {len(trades)}
- Complete Trade Pairs: {total_complete_trades}
- Win Rate: {win_rate*100:.2f}%
- Avg Daily Return: {np.mean(daily_returns)*100:.2f}%
- Transaction Cost: {transaction_cost*100:.2f}%

INTERPRETATION:
- Return > 0% indicates profitable strategy
- Sharpe > 1.0 indicates good risk-adjusted returns
- Max Drawdown < 20% indicates manageable risk
- Win Rate > 50% indicates more winners than losers
"""
    
    with open('../reports/backtest_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Report saved")
    
    print("\n" + "="*70)
    print("✅ BACKTESTING COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()