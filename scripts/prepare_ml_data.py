# ==========================================
# PREPARE DATA FOR MACHINE LEARNING
# Purpose: Split data into train/test and scale features
# Date: 13th November 2025
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("="*60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("="*60)

try:
    # ---- STEP 1: Load labeled data ----
    df = pd.read_csv('../data/labeled_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} samples")
    
    # ---- STEP 2: Select features for the model ----
    # Features = input variables (what model will learn from)
    # These are the technical indicators we calculated
    
    feature_columns = [
        'close',           # Current closing price
        'SMA_5',           # 5-day moving average
        'SMA_20',          # 20-day moving average
        'Momentum_5',      # 5-day price momentum
        'Momentum_10',     # 10-day price momentum
        'Volatility_20',   # 20-day volatility
        'Volume_Ratio'     # Current volume ratio
    ]
    
    # Create X (features) and y (labels)
    X = df[feature_columns]      # Features (inputs)
    y = df['Label']              # Labels (outputs/targets)
    
    print(f"✓ Selected {len(feature_columns)} features")
    print(f"  Features: {feature_columns}")
    
    # ---- STEP 3: Train/Test Split ----
    # Split data into:
    # - 70% for training (model learns from this)
    # - 30% for testing (model is evaluated on this - unseen data)
    # random_state=42 makes it reproducible
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,           # 30% test, 70% train
        random_state=42,         # Reproducible split
        stratify=y               # Keep class ratios same in train/test
    )
    
    print(f"\n✓ Data split into train/test:")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # ---- STEP 4: Check class distribution in splits ----
    train_class_0 = (y_train == 0).sum()
    train_class_1 = (y_train == 1).sum()
    test_class_0 = (y_test == 0).sum()
    test_class_1 = (y_test == 1).sum()
    
    print(f"\n✓ Training set class distribution:")
    print(f"  Class 0: {train_class_0} ({train_class_0/len(y_train)*100:.1f}%)")
    print(f"  Class 1: {train_class_1} ({train_class_1/len(y_train)*100:.1f}%)")
    
    print(f"\n✓ Testing set class distribution:")
    print(f"  Class 0: {test_class_0} ({test_class_0/len(y_test)*100:.1f}%)")
    print(f"  Class 1: {test_class_1} ({test_class_1/len(y_test)*100:.1f}%)")
    
    # ---- STEP 5: Normalize/Scale Features ----
    # Why scale? ML models perform better with normalized features
    # StandardScaler converts data to mean=0, std=1
    # This puts all features on the same scale
    
    # Example: close (2800) vs momentum (0.02) - very different scales!
    # After scaling, both will be between -3 and +3
    
    scaler = StandardScaler()
    
    # Fit scaler on training data only!
    # This is important - we learn scaling parameters from training set
    # Then apply same scaling to test set
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Scaled features using StandardScaler")
    
    # ---- STEP 6: Display scaling statistics ----
    print(f"\nFeature statistics BEFORE scaling:")
    print(X_train[['close', 'Volume_Ratio']].describe())
    
    print(f"\nFeature statistics AFTER scaling:")
    scaled_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    print(scaled_df[['close', 'Volume_Ratio']].describe())
    
    # ---- STEP 7: Save processed data ----
    # Save train/test splits as CSV for later use
    
    # Create DataFrames with features + labels
    train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    train_df['Label'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    test_df['Label'] = y_test.values
    
    train_df.to_csv('../data/train_data_scaled.csv', index=False)
    test_df.to_csv('../data/test_data_scaled.csv', index=False)
    
    print(f"\n✓ Saved train data: ../data/train_data_scaled.csv")
    print(f"✓ Saved test data: ../data/test_data_scaled.csv")
    
    # ---- STEP 8: Save scaler object ----
    # Save scaler so we can use same scaling for future predictions
    
    scaler_file = '../models/scaler.pkl'
    pickle.dump(scaler, open(scaler_file, 'wb'))
    
    print(f"✓ Saved scaler: ../models/scaler.pkl")
    
    # ---- STEP 9: Create info file ----
    # Save information about splits for reference
    
    info = f"""
DATA PREPARATION SUMMARY
========================
Date: {pd.Timestamp.now()}

Dataset Information:
  Total samples: {len(df)}
  Total features: {len(feature_columns)}
  Feature names: {feature_columns}

Train/Test Split:
  Training samples: {len(X_train)} (70%)
  Testing samples: {len(X_test)} (30%)

Training Set:
  Class 0: {train_class_0} ({train_class_0/len(y_train)*100:.1f}%)
  Class 1: {train_class_1} ({train_class_1/len(y_train)*100:.1f}%)

Testing Set:
  Class 0: {test_class_0} ({test_class_0/len(y_test)*100:.1f}%)
  Class 1: {test_class_1} ({test_class_1/len(y_test)*100:.1f}%)

Feature Scaling:
  Method: StandardScaler
  Mean: 0, Std: 1
  Fit on: Training data only

Files Created:
  - ../data/train_data_scaled.csv
  - ../data/test_data_scaled.csv
  - ../models/scaler.pkl
"""
    
    with open('../reports/data_preparation_info.txt', 'w') as f:
        f.write(info)
    
    print(f"\n✓ Saved info: ../reports/data_preparation_info.txt")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print("✅ Ready for Machine Learning!")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
