# ==========================================
# TECHNICAL INDICATORS & FEATURE ENGINEERING
# Purpose: Calculate technical features from stock prices
# Date: 13th November 2025
# ==========================================

import pandas as pd
import numpy as np

print("="*60)
print("CALCULATING TECHNICAL FEATURES")
print("="*60)

try:
    # ---- STEP 1: Load data ----
    # Try to load real data first, fall back to sample
    try:
        df = pd.read_csv('../data/RELIANCE_real_data.csv')
        print("✓ Loaded real data from Zerodha")
    except:
        df = pd.read_csv('../data/RELIANCE_sample.csv')
        print("✓ Loaded sample data")
    
    print(f"✓ Loaded {len(df)} rows of data")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date (oldest first)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # ---- STEP 2: Calculate Simple Moving Average (SMA) ----
    # SMA = average of last N closing prices
    # rolling(window=5) = use last 5 rows
    # mean() = calculate average
    
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    # SMA_5: Average of last 5 days
    # First 4 rows will be NaN (not enough data)
    
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    # SMA_20: Average of last 20 days
    # First 19 rows will be NaN
    
    print("✓ Calculated SMA (5 and 20 periods)")
    
    # ---- STEP 3: Calculate Price Momentum ----
    # Momentum = (Current Price / Price N periods ago) - 1
    # .pct_change() = percentage change from previous value
    
    df['Momentum_5'] = df['close'].pct_change(periods=5)
    # How much price changed over last 5 days (as percentage)
    # Example: 0.02 means +2%, -0.01 means -1%
    
    df['Momentum_10'] = df['close'].pct_change(periods=10)
    # How much price changed over last 10 days
    
    print("✓ Calculated Momentum (5 and 10 periods)")
    
    # ---- STEP 4: Calculate Volatility ----
    # Volatility = standard deviation of returns
    # std() = standard deviation
    # rolling(window=20) = calculate over last 20 days
    
    df['Volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    # How much returns fluctuate over 20 days
    # High volatility = risky, big swings
    # Low volatility = stable, small swings
    
    print("✓ Calculated Volatility")
    
    # ---- STEP 5: Calculate Volume Ratio ----
    # Volume Ratio = Current Volume / Average Volume
    # rolling(window=20) = average over last 20 days
    
    df['Volume_Avg_20'] = df['volume'].rolling(window=20).mean()
    # Average volume over last 20 days
    
    df['Volume_Ratio'] = df['volume'] / df['Volume_Avg_20']
    # How many times above/below average
    # 2.0 = 2x above average
    # 0.5 = half of average
    
    print("✓ Calculated Volume Ratio")
    
    # ---- STEP 6: Calculate Additional Indicators ----
    
    # High-Low Range: Difference between high and low
    df['HL_Range'] = (df['high'] - df['low']) / df['close']
    # Shows volatility during the day
    
    # Price Position: Where close is between high and low
    df['Close_Position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    # 0 = at low, 1 = at high, 0.5 = middle
    
    # Returns: Daily percentage change
    df['Daily_Return'] = df['close'].pct_change()
    # How much price changed from previous day
    
    print("✓ Calculated additional features")
    
    # ---- STEP 7: Remove rows with NaN values ----
    # First 20 rows have NaN because we need 20 days for calculations
    # We'll use .dropna() to remove them
    
    original_rows = len(df)
    df = df.dropna()
    removed_rows = original_rows - len(df)
    
    print(f"\n✓ Removed {removed_rows} rows with NaN values")
    print(f"✓ Remaining rows: {len(df)}")
    
    # ---- STEP 8: Display results ----
    print("\n" + "="*60)
    print("CALCULATED FEATURES")
    print("="*60)
    
    print("\nFirst 5 rows with features:")
    print(df.head())
    
    print("\nFeature Statistics:")
    print(df[['close', 'SMA_5', 'SMA_20', 'Momentum_5', 'Volatility_20', 'Volume_Ratio']].describe())
    
    # ---- STEP 9: Data quality check ----
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Check for NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.sum() == 0:
        print("✓ No missing values")
    else:
        print("✗ Found missing values:")
        print(nan_counts[nan_counts > 0])
    
    # Check for infinite values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_mask.sum() == 0:
        print("✓ No infinite values")
    else:
        print("✗ Found infinite values")
    
    # ---- STEP 10: Save processed data ----
    output_filename = '../data/data_with_features.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Saved to: {output_filename}")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Features created: {len(df.columns)}")
    print(f"Samples available: {len(df)}")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
