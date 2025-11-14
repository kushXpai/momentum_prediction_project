# ==========================================
# DOWNLOAD HISTORICAL STOCK DATA
# Purpose: Fetch 1 month of historical data for RELIANCE
# Date: 13th November 2025
# ==========================================

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")

print("="*60)
print("STOCK DATA DOWNLOAD SCRIPT")
print("="*60)

try:
    # ---- STEP 1: Load saved access token ----
    # Instead of logging in again, we use the token saved earlier
    with open('../config/access_token.txt', 'r') as f:
        access_token = f.read().strip()
    
    # ---- STEP 2: Initialize KiteConnect with saved token ----
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)
    print("✓ Loaded saved session")
    
    # ---- STEP 3: Get list of instruments ----
    # Fetch all available stocks on NSE
    print("\nFetching list of NSE instruments...")
    instruments = kite.instruments("NSE")
    print(f"✓ Found {len(instruments)} stocks")
    
    # ---- STEP 4: Find RELIANCE stock token ----
    # Each stock has a unique token (ID) that we need for API calls
    reliance_token = None
    for instrument in instruments:
        if instrument['tradingsymbol'] == 'RELIANCE':
            reliance_token = instrument['instrument_token']
            break
    
    if reliance_token:
        print(f"✓ Found RELIANCE with token: {reliance_token}")
    else:
        print("✗ Could not find RELIANCE stock")
        exit()
    
    # ---- STEP 5: Set date range ----
    # Download 1 month of data (from 30 days ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nDownloading data from {start_date.date()} to {end_date.date()}")
    
    # ---- STEP 6: Download historical data ----
    # This gets OHLCV data (Open, High, Low, Close, Volume)
    historical_data = kite.historical_data(
        instrument_token=reliance_token,
        from_date=start_date,
        to_date=end_date,
        interval='day'  # Daily bars (other options: 'minute', '5minute', '15minute')
    )
    
    print(f"✓ Downloaded {len(historical_data)} days of data")
    
    # ---- STEP 7: Convert to DataFrame ----
    # Convert the data to a table (DataFrame) for easier manipulation
    df = pd.DataFrame(historical_data)
    
    # ---- STEP 8: Display first few rows ----
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    print("\nLast 5 rows of data:")
    print(df.tail())
    
    # ---- STEP 9: Get data statistics ----
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nClosing Price Statistics:")
    print(f"  Minimum: ₹{df['close'].min():.2f}")
    print(f"  Maximum: ₹{df['close'].max():.2f}")
    print(f"  Average: ₹{df['close'].mean():.2f}")
    print(f"  Last Close: ₹{df['close'].iloc[-1]:.2f}")
    
    print(f"\nVolume Statistics:")
    print(f"  Total volume: {df['volume'].sum():,.0f}")
    print(f"  Average volume: {df['volume'].mean():,.0f}")
    
    # ---- STEP 10: Save to CSV file ----
    # Save the data locally so we don't need to download again
    filename = '../data/RELIANCE_1month.csv'
    df.to_csv(filename, index=False)
    print(f"\n✓ Data saved to: {filename}")
    
    # ---- STEP 11: Display column information ----
    print(f"\nColumn Information:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    print("\nPossible reasons:")
    print("1. Access token has expired (run zerodha_connect.py again)")
    print("2. No internet connection")
    print("3. Zerodha server is down")
    print("4. API rate limit exceeded")