# ==========================================
# HAMNet - HIERARCHICAL ATTENTION MOMENTUM NETWORK
# Purpose: Advanced multi-branch deep learning
# Date: 15th November 2025
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*70)
print("TRAINING HAMNET (HIERARCHICAL ATTENTION MOMENTUM NETWORK)")
print("="*70)

try:
    # ---- STEP 1: Prepare data with multi-modal separation ----
    print("\n[1/7] Preparing multi-modal data...")
    
    train_df = pd.read_csv('../data/train_data_scaled.csv')
    test_df = pd.read_csv('../data/test_data_scaled.csv')
    
    # Define feature groups (modalities)
    # Different types of information
    price_features = ['close']                          # Price movement
    momentum_features = ['Momentum_5', 'Momentum_10']  # Trend indicators
    volatility_features = ['Volatility_20']             # Risk indicators
    volume_features = ['SMA_5', 'SMA_20', 'Volume_Ratio']  # Market activity
    
    # Extract features for each modality
    X_price_train = train_df[price_features].values
    X_momentum_train = train_df[momentum_features].values
    X_volatility_train = train_df[volatility_features].values
    X_volume_train = train_df[volume_features].values
    y_train = train_df['Label'].values
    
    X_price_test = test_df[price_features].values
    X_momentum_test = test_df[momentum_features].values
    X_volatility_test = test_df[volatility_features].values
    X_volume_test = test_df[volume_features].values
    y_test = test_df['Label'].values
    
    print(f"✓ Price features: {X_price_train.shape}")
    print(f"✓ Momentum features: {X_momentum_train.shape}")
    print(f"✓ Volatility features: {X_volatility_train.shape}")
    print(f"✓ Volume features: {X_volume_train.shape}")
    
    # ---- STEP 2: Build HAMNet with multiple inputs ----
    print("\n[2/7] Building HAMNet architecture...")
    
    # Define input layers for each modality
    # Each input expects different shaped data
    
    input_price = layers.Input(shape=(1,), name='price_input')
    input_momentum = layers.Input(shape=(2,), name='momentum_input')
    input_volatility = layers.Input(shape=(1,), name='volatility_input')
    input_volume = layers.Input(shape=(3,), name='volume_input')
    
    print("✓ Input layers created")
    
    # ---- STEP 3: Create price branch ----
    print("\n[3/7] Creating price branch...")
    
    # Price branch processes price data
    x_price = layers.Dense(16, activation='relu')(input_price)
    x_price = layers.Dropout(0.2)(x_price)
    x_price = layers.Dense(8, activation='relu')(x_price)
    
    # ---- STEP 4: Create momentum branch ----
    print("[4/7] Creating momentum branch...")
    
    # Momentum branch processes trend data
    x_momentum = layers.Dense(16, activation='relu')(input_momentum)
    x_momentum = layers.Dropout(0.2)(x_momentum)
    x_momentum = layers.Dense(8, activation='relu')(x_momentum)
    
    # ---- STEP 5: Create volatility branch ----
    print("[5/7] Creating volatility branch...")
    
    # Volatility branch processes risk data
    x_volatility = layers.Dense(16, activation='relu')(input_volatility)
    x_volatility = layers.Dropout(0.2)(x_volatility)
    x_volatility = layers.Dense(8, activation='relu')(x_volatility)

    
    # ---- STEP 6: Create volume branch ----
    print("[6/7] Creating volume branch...")
    
    # Volume branch processes market activity data
    x_volume = layers.Dense(16, activation='relu')(input_volume)
    x_volume = layers.Dropout(0.2)(x_volume)
    x_volume = layers.Dense(8, activation='relu')(x_volume)
    
    # ---- STEP 7: Merge branches and create decision layer ----
    print("\n[7/7] Merging branches with attention...")
    
    # Concatenate all branch outputs
    # This combines all modalities into single vector
    merged = layers.Concatenate()([x_price, x_momentum, x_volatility, x_volume])
    
    # Hidden layers after merge
    merged = layers.Dense(32, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(16, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    
    # Output layer: single neuron with sigmoid for binary classification
    output = layers.Dense(1, activation='sigmoid', name='output')(merged)
    
    # Create model with multiple inputs
    model = keras.Model(
        inputs=[input_price, input_momentum, input_volatility, input_volume],
        outputs=output,
        name='HAMNet'
    )
    
    print("✓ HAMNet architecture created")
    
    # ---- STEP 8: Compile model ----
    print("\n[8/8] Compiling model...")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("✓ Model compiled")
    
    # Print model summary
    print("\nModel Summary:")
    print("-" * 70)
    model.summary()
    
    # ---- STEP 9: Train HAMNet ----
    print("\nTraining HAMNet...")
    print("(This may take 2-3 minutes)")
    print("-" * 70)
    
    history = model.fit(
        [X_price_train, X_momentum_train, X_volatility_train, X_volume_train],
        y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        verbose=1,
        shuffle=True
    )
    
    print("\n✓ Training complete")
    
    # ---- STEP 10: Evaluate on test set ----
    print("\nEvaluating on test set...")
    
    y_pred_proba = model.predict(
        [X_price_test, X_momentum_test, X_volatility_test, X_volume_test],
        verbose=0
    )
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*70)
    print("HAMNET PERFORMANCE")
    print("="*70)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # ---- STEP 11: Save model ----
    print("\nSaving model...")
    model.save('../models/hamnet_model.keras')
    print("✓ HAMNet saved to: ../models/hamnet_model.h5")
    
    # ---- STEP 12: Plot results ----
    print("\nPlotting results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy subplot
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('HAMNet Accuracy Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss subplot
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('HAMNet Loss Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/hamnet_training_history.png', dpi=100)
    print("✓ HAMNet training history saved")
    plt.show()
    
    print("\n" + "="*70)
    print("✅ HAMNET TRAINING COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()