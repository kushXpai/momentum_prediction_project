# ==========================================
# CREATE TRADING LABELS
# Purpose: Label data as "momentum" or "not momentum"
# Date: 13th November 2025
# ==========================================

import pandas as pd
import numpy as np

print("="*60)
print("CREATING TRAINING LABELS")
print("="*60)

try:
    # ---- STEP 1: Load features ----
    df = pd.read_csv('../data/data_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} rows")
    
    # ---- STEP 2: Calculate future returns ----
    # We need to know what price will be in 5 days
    # to determine if there's momentum
    
    # shift(-5) = look 5 rows ahead
    # This gets the closing price from 5 days in the future
    
    df['Future_Close'] = df['close'].shift(-5)
    # Future_Close: What the price will be in 5 days
    # First row's future_close = 6th row's close
    # Last 5 rows = NaN (no future data)
    
    print("✓ Calculated future prices (5-day forward look)")
    
    # ---- STEP 3: Calculate future returns ----
    # Return = (Future Price - Current Price) / Current Price
    # This shows percentage change from today to 5 days later
    
    df['Future_Return'] = (df['Future_Close'] - df['close']) / df['close']
    # Positive return = price will go up
    # Negative return = price will go down
    # Example: 0.02 = +2% return, -0.01 = -1% return
    
    print("✓ Calculated future returns")
    
    # ---- STEP 4: Create binary labels ----
    # Define "momentum" as +2% or more gain in 5 days
    # We could use any threshold (1%, 3%, 5%)
    # 2% is a reasonable middle ground
    
    momentum_threshold = 0.02  # 2% threshold
    
    # Create label: 1 if return > 2%, else 0
    df['Label'] = (df['Future_Return'] > momentum_threshold).astype(int)
    # .astype(int) converts True/False to 1/0
    # Label = 1: Good opportunity to buy (high momentum expected)
    # Label = 0: Not a good opportunity (low or negative momentum)
    
    print(f"✓ Created labels (threshold: {momentum_threshold*100}%)")
    
    # ---- STEP 5: Remove rows with NaN ----
    # Last 5 rows have NaN because no future data
    # Also earlier rows with NaN from feature engineering
    
    original_length = len(df)
    df = df.dropna()
    removed = original_length - len(df)
    
    print(f"✓ Removed {removed} rows with NaN")
    print(f"✓ Final dataset: {len(df)} rows")
    
    # ---- STEP 6: Analyze class distribution ----
    # Check how many samples of each class
    # (ideally should be balanced - roughly equal 0s and 1s)
    
    class_0_count = (df['Label'] == 0).sum()
    class_1_count = (df['Label'] == 1).sum()
    total = len(df)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"  Class 0 (No Momentum): {class_0_count} ({class_0_count/total*100:.1f}%)")
    print(f"  Class 1 (Momentum): {class_1_count} ({class_1_count/total*100:.1f}%)")
    
    # Imbalance ratio
    if class_1_count > 0:
        imbalance_ratio = class_0_count / class_1_count
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("  ⚠️  Warning: Data is imbalanced!")
            print("     This may require special handling")
    
    # ---- STEP 7: Display samples of each class ----
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    
    print("\nSamples with Momentum (Label=1):")
    momentum_samples = df[df['Label'] == 1].head(3)
    print(momentum_samples[['date', 'close', 'Future_Return', 'Label']])
    
    print("\nSamples without Momentum (Label=0):")
    no_momentum_samples = df[df['Label'] == 0].head(3)
    print(no_momentum_samples[['date', 'close', 'Future_Return', 'Label']])
    
    # ---- STEP 8: Statistics ----
    print("\n" + "="*60)
    print("RETURN STATISTICS")
    print("="*60)
    print(f"Mean return: {df['Future_Return'].mean()*100:.2f}%")
    print(f"Std deviation: {df['Future_Return'].std()*100:.2f}%")
    print(f"Min return: {df['Future_Return'].min()*100:.2f}%")
    print(f"Max return: {df['Future_Return'].max()*100:.2f}%")
    print(f"Median return: {df['Future_Return'].median()*100:.2f}%")
    
    # ---- STEP 9: Save labeled data ----
    output_file = '../data/labeled_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved labeled data to: {output_file}")
    
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()