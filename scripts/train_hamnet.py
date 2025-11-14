# ==========================================
# HAMNet - HIERARCHICAL ATTENTION MOMENTUM NETWORK
# Purpose: Advanced multi-branch deep learning
# Date: 15th November 2025
# ==========================================

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

print("="*70)
print("TRAINING HAMNET (HIERARCHICAL ATTENTION MOMENTUM NETWORK)")
print("="*70)

try:
    # ---- STEP 1: Load data ----
    print("\n[1/8] Loading labeled data...")
    
    df = pd.read_csv('../data/labeled_data.csv')
    print(f"✓ Loaded {len(df)} samples")
    
    # ---- STEP 2: Prepare features and labels ----
    print("\n[2/8] Preparing features and labels...")
    
    feature_columns = ['close', 'SMA_5', 'SMA_20', 'Momentum_5', 
                      'Momentum_10', 'Volatility_20', 'Volume_Ratio']
    
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Labels distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # ---- STEP 3: Scale features ----
    print("\n[3/8] Scaling features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pickle.dump(scaler, open('../models/scaler.pkl', 'wb'))
    print("✓ Features scaled")
    
    # ---- STEP 4: Split data ----
    print("\n[4/8] Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ---- STEP 5: Build HAMNet model ----
    print("\n[5/8] Building HAMNet architecture...")
    
    # Input layer
    input_layer = layers.Input(shape=(7,), name='input')
    
    # Price branch (close, SMA_5, SMA_20)
    price_input = layers.Lambda(lambda x: x[:, :3])(input_layer)
    price_branch = layers.Dense(32, activation='relu')(price_input)
    price_branch = layers.Dense(16, activation='relu')(price_branch)
    
    # Momentum branch (Momentum_5, Momentum_10)
    momentum_input = layers.Lambda(lambda x: x[:, 3:5])(input_layer)
    momentum_branch = layers.Dense(32, activation='relu')(momentum_input)
    momentum_branch = layers.Dense(16, activation='relu')(momentum_branch)
    
    # Volatility branch (Volatility_20)
    volatility_input = layers.Lambda(lambda x: x[:, 5:6])(input_layer)
    volatility_branch = layers.Dense(16, activation='relu')(volatility_input)
    volatility_branch = layers.Dense(8, activation='relu')(volatility_branch)
    
    # Volume branch (Volume_Ratio)
    volume_input = layers.Lambda(lambda x: x[:, 6:7])(input_layer)
    volume_branch = layers.Dense(16, activation='relu')(volume_input)
    volume_branch = layers.Dense(8, activation='relu')(volume_branch)
    
    # Fusion layer - concatenate all branches
    fusion = layers.Concatenate()([price_branch, momentum_branch, volatility_branch, volume_branch])
    
    # Attention-like mechanism
    fusion = layers.Dense(64, activation='relu')(fusion)
    fusion = layers.Dropout(0.3)(fusion)
    fusion = layers.Dense(32, activation='relu')(fusion)
    fusion = layers.Dropout(0.2)(fusion)
    fusion = layers.Dense(16, activation='relu')(fusion)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output, name='HAMNet')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ HAMNet model created")
    print(model.summary())
    
    # ---- STEP 6: Train HAMNet ----
    print("\n[6/8] Training HAMNet...")
    print("   (This may take 2-3 minutes)")
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    print("✓ Training complete")
    
    # ---- STEP 7: Evaluate HAMNet ----
    print("\n[7/8] Evaluating HAMNet...")
    
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
    
    # ---- STEP 8: Save model ----
    print("\n[8/8] Saving model...")
    
    model.save('../models/hamnet_model.h5')
    print("✓ HAMNet saved to: ../models/hamnet_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('HAMNet: Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('HAMNet: Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../reports/hamnet_training_history.png', dpi=150)
    print("✓ Training plot saved")
    
    print("\n" + "="*70)
    print("✅ HAMNET TRAINING COMPLETE")
    print("="*70)
    print(f"\nHAMNet Performance:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1 Score: {f1:.4f}")

except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()