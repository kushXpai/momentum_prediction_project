# ==========================================
# SIMPLE NEURAL NETWORK FOR MOMENTUM PREDICTION
# Purpose: Build first deep learning model
# Date: 15th November 2025
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("="*70)
print("TRAINING SIMPLE NEURAL NETWORK")
print("="*70)

try:
    # ---- STEP 1: Load training data ----
    print("\n[1/8] Loading data...")
    
    train_df = pd.read_csv('../data/train_data_scaled.csv')
    test_df = pd.read_csv('../data/test_data_scaled.csv')
    
    # Separate features (X) from labels (y)
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label'].values
    
    print(f"✓ Training data: {X_train.shape} samples, {X_train.shape} features")
    print(f"✓ Testing data: {X_test.shape} samples")
    
    # ---- STEP 2: Build neural network architecture ----
    print("\n[2/8] Building neural network...")
    
    # Sequential model = layers stacked one on top of another
    model = keras.Sequential([
        # Layer 1: Input layer
        # 32 neurons (units)
        # relu = activation function (adds non-linearity)
        # input_shape=(7,) = expects 7 features as input
        keras.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),

        
        # Dropout layer: randomly deactivates 30% of neurons
        # Purpose: prevent overfitting
        layers.Dropout(0.3),
        
        # Layer 2: 16 neurons
        # Each neuron takes input from previous layer (32 neurons)
        layers.Dense(16, activation='relu'),
        
        # Another dropout layer
        layers.Dropout(0.3),
        
        # Layer 3: 8 neurons
        layers.Dense(8, activation='relu'),
        
        # Output layer: 1 neuron with sigmoid
        # sigmoid = converts output to probability (0-1)
        # 1 neuron because we have binary classification (0 or 1)
        layers.Dense(1, activation='sigmoid')
    ])
    
    print("✓ Model architecture created")
    
    # ---- STEP 3: Compile model ----
    print("\n[3/8] Compiling model...")
    
    # Optimizer: adam = optimization algorithm to update weights
    # Loss: binary_crossentropy = for binary classification
    # Metrics: accuracy = what to track during training
    
    model.compile(
        optimizer='adam',                    # Optimization algorithm
        loss='binary_crossentropy',          # Loss function for binary classification
        metrics=['accuracy', 'precision', 'recall']  # Track these during training
    )
    
    print("✓ Model compiled")
    
    # ---- STEP 4: Print model summary ----
    print("\n[4/8] Model Summary:")
    print("-" * 70)
    model.summary()
    
    # ---- STEP 5: Train the model ----
    print("\n[5/8] Training neural network...")
    print("(This may take 1-3 minutes)")
    print("-" * 70)
    
    # Train the model
    # epochs = number of times to go through entire dataset
    # batch_size = how many samples to process before updating weights
    # validation_split = use 20% of training data for validation
    # verbose = 1 shows progress bar
    
    history = model.fit(
        X_train, y_train,
        epochs=50,                    # Train for 50 iterations
        batch_size=8,                 # Process 8 samples at a time
        validation_split=0.2,         # 20% for validation, 80% for training
        verbose=1,                    # Show progress
        shuffle=True                  # Randomize order each epoch
    )
    
    print("\n✓ Training complete")
    
    # ---- STEP 6: Make predictions ----
    print("\n[6/8] Making predictions...")
    
    # Get probability predictions (values between 0 and 1)
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Convert probabilities to binary predictions (0 or 1)
    # If probability > 0.5, predict 1, else predict 0
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    print("✓ Predictions complete")
    
    # ---- STEP 7: Evaluate model ----
    print("\n[7/8] Evaluating model...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("\n" + "="*70)
    print("NEURAL NETWORK PERFORMANCE")
    print("="*70)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {int(tn)}")
    print(f"  False Positives: {int(fp)}")
    print(f"  False Negatives: {int(fn)}")
    print(f"  True Positives:  {int(tp)}")
    
    # ---- STEP 8: Save model ----
    print("\n[8/8] Saving model...")
    
    model.save('../models/simple_nn_model.keras')
    print("✓ Model saved to: ../models/simple_nn_model.h5")
    
    # ---- STEP 9: Plot training history ----
    print("\nPlotting training history...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy over epochs
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot loss over epochs
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/nn_training_history.png', dpi=100)
    print("✓ Training history saved to: ../reports/nn_training_history.png")
    # plt.show()
    
    print("\n" + "="*70)
    print("✅ NEURAL NETWORK TRAINING COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()