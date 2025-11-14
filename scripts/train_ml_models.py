# ==========================================
# TRAIN MULTIPLE MACHINE LEARNING MODELS
# Purpose: Train and compare different ML algorithms
# Date: 13th November 2025
# ==========================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING MACHINE LEARNING MODELS")
print("="*70)

try:
    # ---- STEP 1: Load training and testing data ----
    print("\n[1/6] Loading data...")
    
    train_df = pd.read_csv('../data/train_data_scaled.csv')
    test_df = pd.read_csv('../data/test_data_scaled.csv')
    
    # Separate features and labels
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']
    
    print(f"✓ Training data: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"✓ Testing data: {len(X_test)} samples")
    
    # ---- STEP 2: Define models ----
    print("\n[2/6] Defining models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,           # Maximum iterations for training
            random_state=42,         # For reproducibility
            solver='lbfgs'           # Algorithm to use
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,        # Number of trees
            max_depth=10,            # Limit tree depth (prevent overfitting)
            random_state=42,
            n_jobs=-1                # Use all CPU cores
        ),
        'SVM': SVC(
            kernel='rbf',            # Kernel type (rbf = non-linear)
            C=1.0,                   # Regularization parameter
            probability=True,        # Enable probability predictions
            random_state=42
        )
    }
    
    print(f"✓ Defined {len(models)} models:")
    for name in models.keys():
        print(f"   - {name}")
    
    # ---- STEP 3: Train models and collect results ----
    print("\n[3/6] Training models...")
    print("-" * 70)
    
    results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        print(f"  ✓ Training complete")
    
    # ---- STEP 4: Make predictions ----
    print("\n[4/6] Making predictions...")
    print("-" * 70)
    
    predictions = {}
    
    for model_name, model in trained_models.items():
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Get probability predictions (for ROC-AUC)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
            # Normalize to 0-1 range
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  ✓ {model_name} predictions complete")
    
    # ---- STEP 5: Evaluate models ----
    print("\n[5/6] Evaluating models...")
    print("-" * 70)
    
    for model_name in predictions.keys():
        y_pred = predictions[model_name]['y_pred']
        y_pred_proba = predictions[model_name]['y_pred_proba']
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
    
    # ---- STEP 6: Display results ----
    print("\n[6/6] Results")
    print("="*70)
    
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 70)
    
    results_df = pd.DataFrame(results).T  # Transpose for better display
    
    # Display each metric
    metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for metric in metrics_to_show:
        print(f"\n{metric}:")
        for model_name in results.keys():
            value = results[model_name][metric]
            print(f"  {model_name:25s}: {value:.4f} ({value*100:.2f}%)")
    
    # ---- STEP 7: Detailed results for each model ----
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for model_name in results.keys():
        r = results[model_name]
        
        print(f"\n{model_name}:")
        print("-" * 70)
        print(f"  Accuracy:   {r['Accuracy']:.4f} (Correct predictions)")
        print(f"  Precision:  {r['Precision']:.4f} (Of predicted 1s, how many correct?)")
        print(f"  Recall:     {r['Recall']:.4f} (Of actual 1s, how many caught?)")
        print(f"  F1-Score:   {r['F1-Score']:.4f} (Balance of precision & recall)")
        print(f"  ROC-AUC:    {r['ROC-AUC']:.4f} (Overall ranking ability)")
        
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {int(r['TN']):3d}  (Correctly said 'No')")
        print(f"    False Positives: {int(r['FP']):3d}  (Wrongly said 'Yes')")
        print(f"    False Negatives: {int(r['FN']):3d}  (Wrongly said 'No')")
        print(f"    True Positives:  {int(r['TP']):3d}  (Correctly said 'Yes')")
    
    # ---- STEP 8: Save best models ----
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    for model_name, model in trained_models.items():
        # Replace spaces with underscores for filename
        filename = f"../models/{model_name.replace(' ', '_')}.pkl"
        pickle.dump(model, open(filename, 'wb'))
        print(f"✓ Saved {model_name} to {filename}")
    
    # ---- STEP 9: Find best model ----
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    
    best_model_name = max(results, key=lambda x: results[x]['ROC-AUC'])
    best_auc = results[best_model_name]['ROC-AUC']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  ROC-AUC: {best_auc:.4f}")
    print(f"\nUse this model for production predictions!")
    
    # ---- STEP 10: Save results summary ----
    results_summary = f"""
MACHINE LEARNING MODELS - TRAINING SUMMARY
=============================================

Models Trained: {len(models)}
Training Samples: {len(X_train)}
Testing Samples: {len(X_test)}
Features Used: {len(X_train.columns)}

BEST MODEL: {best_model_name}
- ROC-AUC: {best_auc:.4f}

ALL RESULTS:
"""
    
    for model_name in results.keys():
        r = results[model_name]
        results_summary += f"""
{model_name}:
  Accuracy: {r['Accuracy']:.4f}
  Precision: {r['Precision']:.4f}
  Recall: {r['Recall']:.4f}
  F1-Score: {r['F1-Score']:.4f}
  ROC-AUC: {r['ROC-AUC']:.4f}
"""
    
    with open('../reports/ml_models_summary.txt', 'w') as f:
        f.write(results_summary)
    
    print(f"\n✓ Saved summary: ../reports/ml_models_summary.txt")
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
