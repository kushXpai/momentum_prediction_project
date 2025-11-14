# ==========================================
# DOWNLOAD REAL STOCK DATA FROM ZERODHA
# Purpose: Fetch real historical data from Zerodha API
# Date: 13th November 2025
# ==========================================

from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")

print("="*70)
print("DOWNLOADING REAL DATA FROM ZERODHA API")
print("="*70)

try:
    # ---- STEP 1: Initialize Zerodha KiteConnect ----
    print("\n[1/7] Initializing KiteConnect...")
    
    kite = KiteConnect(api_key=API_KEY)
    print("✓ KiteConnect initialized")
    
    # ---- STEP 2: Load or generate access token ----
    print("\n[2/7] Setting up access token...")
    
    try:
        # Try to load saved access token
        with open('../config/access_token.txt', 'r') as f:
            access_token = f.read().strip()
        
        kite.set_access_token(access_token)
        print("✓ Access token loaded from file")
    
    except FileNotFoundError:
        print("✗ No saved access token found")
        print("\nGenerating new access token...")
        print("Follow these steps:")
        print("1. Visit the login URL")
        print("2. Login with your Zerodha credentials")
        print("3. Copy the 'request_token' from the redirect URL")
        
        # Generate login URL
        login_url = kite.login_url()
        print(f"\nLogin URL:\n{login_url}\n")
        
        # Get request token from user
        request_token = input("Paste the request_token here: ").strip()
        
        # Generate session
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        
        # Save access token for future use
        with open('../config/access_token.txt', 'w') as f:
            f.write(access_token)
        
        kite.set_access_token(access_token)
        print("✓ Access token generated and saved")
    
    # ---- STEP 3: Get instrument token for RELIANCE ----
    print("\n[3/7] Finding RELIANCE instrument token...")
    
    # Fetch all NSE instruments
    instruments = kite.instruments("NSE")
    
    # Find RELIANCE
    reliance_token = None
    for instrument in instruments:
        if instrument['tradingsymbol'] == 'RELIANCE':
            reliance_token = instrument['instrument_token']
            print(f"✓ RELIANCE token found: {reliance_token}")
            break
    
    if not reliance_token:
        raise Exception("RELIANCE instrument not found")
    
    # ---- STEP 4: Define date range ----
    print("\n[4/7] Setting date range...")
    
    # Get last 100 trading days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)  # Extra buffer for weekends
    
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    
    # ---- STEP 5: Fetch historical data ----
    print("\n[5/7] Fetching historical data from Zerodha...")
    print("(This may take 10-30 seconds)")
    
    # Fetch daily OHLCV data
    historical_data = kite.historical_data(
        instrument_token=reliance_token,
        from_date=start_date,
        to_date=end_date,
        interval='day'
    )
    
    print(f"✓ Downloaded {len(historical_data)} candles")
    
    # ---- STEP 6: Convert to DataFrame ----
    print("\n[6/7] Processing data...")
    
    df = pd.DataFrame(historical_data)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    # Keep only required columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Display summary
    print(f"\nData Summary:")
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
    print(f"  Current price: ₹{df['close'].iloc[-1]:.2f}")
    
    # Display first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # ---- STEP 7: Save to CSV ----
    print("\n[7/7] Saving data...")
    
    # Save to data folder
    output_file = '../data/RELIANCE_real_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Data saved to: {output_file}")
    
    # ---- ADDITIONAL: Save metadata ----
    metadata = {
        'stock': 'RELIANCE',
        'instrument_token': reliance_token,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'total_candles': len(df),
        'download_timestamp': datetime.now().isoformat(),
        'data_source': 'Zerodha Kite Connect API'
    }
    
    metadata_file = '../data/RELIANCE_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_file}")
    
    # ---- SUCCESS SUMMARY ----
    print("\n" + "="*70)
    print("✅ REAL DATA DOWNLOAD COMPLETE")
    print("="*70)
    
    print(f"\nFiles created:")
    print(f"  1. {output_file}")
    print(f"  2. {metadata_file}")
    
    print(f"\nData statistics:")
    print(f"  Stock: RELIANCE")
    print(f"  Days: {len(df)}")
    print(f"  Latest price: ₹{df['close'].iloc[-1]:.2f}")
    print(f"  Average volume: {df['volume'].mean():,.0f}")
    
    print(f"\n✓ Ready for feature engineering (next step)")

except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check API_KEY and API_SECRET are correct")
    print("2. Verify access token is valid")
    print("3. Ensure internet connection is stable")
    print("4. Check if market data subscription is active")
    print("5. Verify Zerodha account status")
    
    import traceback
    traceback.print_exc()