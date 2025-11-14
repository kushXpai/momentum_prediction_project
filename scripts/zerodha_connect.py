# ==========================================
# ZERODHA API CONNECTION SCRIPT
# Purpose: Test connection to Zerodha API
# Date: 13th November 2025
# ==========================================

# Import the KiteConnect library (from kiteconnect package)
# This library handles all communication with Zerodha
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os

load_dotenv()

# ---- STEP 1: Set up API credentials ----
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")

print("="*50)
print("ZERODHA API CONNECTION TEST")
print("="*50)

try:
    # ---- STEP 2: Initialize KiteConnect object ----
    # This creates a connection object that will communicate with Zerodha
    kite = KiteConnect(api_key=API_KEY)
    print("✓ KiteConnect initialized")
    
    # ---- STEP 3: Generate login URL ----
    # This creates a URL where user can login to Zerodha
    login_url = kite.login_url()
    print(f"\n✓ Login URL generated")
    print(f"Visit this URL to login:\n{login_url}")
    
    # ---- STEP 4: Get request token from user ----
    # After logging in on the URL above, user gets a request_token
    # We need this to generate a session
    print("\n" + "="*50)
    print("After logging in on the URL above:")
    print("1. You'll be redirected to a page")
    print("2. Copy the 'request_token' from the URL")
    print("3. Paste it below")
    print("="*50)
    
    request_token = input("\nEnter your request_token: ")
    
    # ---- STEP 5: Generate session ----
    # This exchanges the request_token for an access_token
    # Access token is used for all future API calls
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    
    # Extract the access token from the response
    access_token = data["access_token"]
    
    # ---- STEP 6: Set the access token ----
    # This tells KiteConnect to use this token for future requests
    kite.set_access_token(access_token)
    
    # ---- STEP 7: Save access token (optional) ----
    # Save token to file for later use (so you don't have to login again)
    with open('../config/access_token.txt', 'w') as f:
        f.write(access_token)
    
    print("\n" + "="*50)
    print("✓ SESSION CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"Access Token saved to: ../config/access_token.txt")
    print(f"Token: {access_token[:20]}...")  # Show first 20 chars
    
    # ---- STEP 8: Test the connection ----
    # Try to fetch list of instruments (stocks available on NSE)
    print("\nFetching list of available stocks (NSE)...")
    instruments = kite.instruments("NSE")
    print(f"✓ Successfully fetched {len(instruments)} stocks")
    print(f"First 5 stocks: {[inst['tradingsymbol'] for inst in instruments[:5]]}")
    
except Exception as e:
    # If anything goes wrong, catch the error and print it
    print(f"\n✗ ERROR: {str(e)}")
    print("Possible reasons:")
    print("1. API credentials are incorrect")
    print("2. Request token has expired")
    print("3. No internet connection")
    print("4. Zerodha server is down")