# ==========================================
# VISUALIZE FEATURES
# Purpose: Create charts showing features
# Date: 13th November 2025
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("VISUALIZING FEATURES")
print("="*60)

try:
    # ---- STEP 1: Load data ----
    df = pd.read_csv('../data/data_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} rows of data")
    
    # ---- STEP 2: Create figure with subplots ----
    # figsize=(15, 12) = figure size 15x12 inches
    # 3 rows, 1 column = 3 charts stacked vertically
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Stock Technical Features Analysis', fontsize=16, fontweight='bold')
    
    # ---- STEP 3: Chart 1 - Price and Moving Averages ----
    axes[0].plot(df['date'], df['close'], label='Close Price', color='blue', linewidth=2)
    axes[0].plot(df['date'], df['SMA_5'], label='SMA 5', color='red', linewidth=1.5, linestyle='--')
    axes[0].plot(df['date'], df['SMA_20'], label='SMA 20', color='green', linewidth=1.5, linestyle='--')
    axes[0].set_title('Price with Moving Averages', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price (₹)', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # ---- STEP 4: Chart 2 - Momentum ----
    axes[1].bar(df['date'], df['Momentum_5'], label='Momentum 5', color='purple', alpha=0.6)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Price Momentum', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Momentum (%)', fontsize=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # ---- STEP 5: Chart 3 - Volume Ratio ----
    axes[2].bar(df['date'], df['Volume_Ratio'], label='Volume Ratio', color='orange', alpha=0.6)
    axes[2].axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Average (1.0)')
    axes[2].set_title('Volume Ratio (Current / Average)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Ratio', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    # ---- STEP 6: Adjust layout and save ----
    plt.tight_layout()
    
    # Save figure
    output_file = '../reports/feature_visualization.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved chart to: {output_file}")
    
    # Show figure
    # plt.show()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc() 