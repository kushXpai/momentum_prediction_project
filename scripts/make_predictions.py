# ==========================================
# MAKE PREDICTIONS ON NEW DATA
# Purpose: Use trained model for predictions
# Date: 13th November 2025
# ==========================================

import pandas as pd
import pickle
import numpy as np

print("="*60)
print("MAKING PREDICTIONS WITH TRAINED MODEL")
print("="*60)

try:
    # ---- STEP 1: Load the best model ----
    # We found Random Forest was best with 0.9259 ROC-AUC
    
    model = pickle.load(open('../models/Random_Forest.pkl', 'rb'))
    scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
    
    print("✓ Loaded Random Forest model")
    print("✓ Loaded scaler")
    
    # ---- STEP 2: Create sample input ----
    # New stock data (must have same 7 features)
    
    sample_data = {
        'close': [2900.50],
        'SMA_5': [2895.00],
        'SMA_20': [2890.00],
        'Momentum_5': [0.0225],
        'Momentum_10': [0.0450],
        'Volatility_20': [0.0185],
        'Volume_Ratio': [1.35]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    print("\nInput Data:")
    print(df_sample)
    
    # ---- STEP 3: Scale input ----
    df_scaled = scaler.transform(df_sample)

    # Wrap back into DataFrame with column names to avoid warnings
    df_scaled_named = pd.DataFrame(df_scaled, columns=df_sample.columns)
    
    print("\n✓ Data scaled using saved scaler")
    
    # ---- STEP 4: Make prediction ----
    prediction = model.predict(df_scaled_named)
    probability = model.predict_proba(df_scaled_named)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    
    print(f"\nPrediction: {'BUY SIGNAL (Momentum Expected)' if prediction == 1 else 'HOLD (No Momentum)'}")
    prob_class1 = probability[0][1]  # first row, class 1
    print(f"Confidence: {prob_class1:.2%}")
    print(f"Signal Strength: ", end="")
    if prob_class1 > 0.8:
        print("🟢 STRONG")
    elif prob_class1 > 0.6:
        print("🟡 MODERATE")
    else:
        print("🔴 WEAK")
    
    print("\n" + "="*60)
    
    # ---- STEP 5: Test on multiple samples ----
    print("\n\nTesting on Multiple Samples:")
    print("-" * 60)
    
    test_cases = [
        {'close': 2850, 'SMA_5': 2845, 'SMA_20': 2840, 'Momentum_5': 0.01, 'Momentum_10': 0.02, 'Volatility_20': 0.015, 'Volume_Ratio': 0.9},
        {'close': 2900, 'SMA_5': 2895, 'SMA_20': 2890, 'Momentum_5': 0.03, 'Momentum_10': 0.05, 'Volatility_20': 0.02, 'Volume_Ratio': 1.5},
        {'close': 2800, 'SMA_5': 2810, 'SMA_20': 2820, 'Momentum_5': -0.01, 'Momentum_10': -0.02, 'Volatility_20': 0.025, 'Volume_Ratio': 1.1},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        df_test = pd.DataFrame([test_case])
        df_test_scaled = scaler.transform(df_test)
        df_test_scaled_named = pd.DataFrame(df_test_scaled, columns=df_test.columns)
        pred = model.predict(df_test_scaled_named)
        prob = model.predict_proba(df_test_scaled_named)
        
        signal = "BUY" if pred == 1 else "HOLD"
        prob_class1 = prob[0][1]  # first row, class 1
        print(f"\nTest {i}:")
        print(f"  Close: ₹{test_case['close']:.2f}")
        print(f"  Signal: {signal} ({prob_class1:.2%} confidence)")
    
    print("\n" + "="*60)
    print("✅ PREDICTIONS COMPLETE")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
