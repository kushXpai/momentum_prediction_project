# ==========================================
# SIMPLE NEURAL NETWORK FOR MOMENTUM PREDICTION
# Purpose: Build first deep learning model
# Date: 15th November 2025
# ==========================================

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

print("="*70)
print("TRAINING SIMPLE NEURAL NETWORK FOR MOMENTUM PREDICTION")
print("="*70)

try:
    # ---- STEP 1: Load data ----
    print("\n[1/7] Loading labeled data...")
    
    df = pd.read_csv('../data/labeled_data.csv')
    print(f"✓ Loaded {len(df)} samples")
    
    # ---- STEP 2: Prepare features and labels ----
    print("\n[2/7] Preparing features and labels...")
    
    feature_columns = ['close', 'SMA_5', 'SMA_20', 'Momentum_5', 
                      'Momentum_10', 'Volatility_20', 'Volume_Ratio']
    
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Labels: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # Check for class balance
    if np.sum(y==0) == 0 or np.sum(y==1) == 0:
        print("⚠️  WARNING: Class imbalance detected!")
        print("   This will affect model training")
    
    # ---- STEP 3: Scale features ----
    print("\n[3/7] Scaling features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for predictions
    pickle.dump(scaler, open('../models/scaler.pkl', 'wb'))
    print("✓ Features scaled and scaler saved")
    
    # ---- STEP 4: Split data ----
    print("\n[4/7] Splitting data (80% train, 20% test)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Train set: {X_train.shape} samples")
    print(f"✓ Test set: {X_test.shape} samples")
    
    # ---- STEP 5: Build Simple NN model ----
    print("\n[5/7] Building Simple Neural Network...")
    
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=7),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model architecture:")
    print(model.summary())
    
    # ---- STEP 6: Train model ----
    print("\n[6/7] Training model...")
    print("   (This may take 1-2 minutes)")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print("✓ Training complete")
    
    # ---- STEP 7: Evaluate model ----
    print("\n[7/7] Evaluating model...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✓ Test Loss: {loss:.4f}")
    print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
    
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✓ Precision: {precision*100:.2f}%")
    print(f"✓ Recall: {recall*100:.2f}%")
    print(f"✓ F1 Score: {f1:.4f}")
    
    # ---- Save model ----
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    model.save('../models/simple_nn_model.h5')
    print("✓ Model saved to: ../models/simple_nn_model.h5")
    
    # ---- Plot training history ----
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Simple NN: Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Simple NN: Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../reports/simple_nn_training_history.png', dpi=150)
    print("✓ Training plot saved")
    
    print("\n" + "="*70)
    print("✅ SIMPLE NN TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1 Score: {f1:.4f}")

except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()