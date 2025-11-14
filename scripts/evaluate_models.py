# ==========================================
# VISUALIZE MODEL PERFORMANCE
# Purpose: Create comparison charts
# Date: 13th November 2025
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("="*60)
print("EVALUATING MODEL PERFORMANCE")
print("="*60)

try:
    # ---- STEP 1: Load test data ----
    test_df = pd.read_csv('../data/test_data_scaled.csv')
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']
    
    print(f"✓ Loaded test data: {len(X_test)} samples")
    
    # ---- STEP 2: Load trained models ----
    models = {
        'Logistic Regression': pickle.load(open('../models/Logistic_Regression.pkl', 'rb')),
        'Random Forest': pickle.load(open('../models/Random_Forest.pkl', 'rb')),
        'SVM': pickle.load(open('../models/SVM.pkl', 'rb'))
    }
    
    print(f"✓ Loaded {len(models)} models")
    
    # ---- STEP 3: Create evaluation figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Machine Learning Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # ---- STEP 4: Collect results ----
    accuracies = []
    precisions = []
    recalls = []
    aucs = []
    model_names = list(models.keys())
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(roc_auc)
    
    # ---- STEP 5: Chart 1 - Model Comparison ----
    x = np.arange(len(model_names))
    width = 0.2
    
    axes[0, 0].bar(x - width*1.5, accuracies, width, label='Accuracy', color='blue')
    axes[0, 0].bar(x - width*0.5, precisions, width, label='Precision', color='green')
    axes[0, 0].bar(x + width*0.5, recalls, width, label='Recall', color='orange')
    axes[0, 0].bar(x + width*1.5, aucs, width, label='ROC-AUC', color='red')
    
    axes[0, 0].set_ylabel('Score', fontsize=10)
    axes[0, 0].set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # ---- STEP 6: Chart 2 - Accuracy Comparison ----
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0, 1].barh(model_names, accuracies, color=colors)
    axes[0, 1].set_xlabel('Accuracy', fontsize=10)
    axes[0, 1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 1].text(v + 0.02, i, f'{v:.2%}', va='center')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # ---- STEP 7: Chart 3 - ROC-AUC Comparison ----
    axes[1, 0].barh(model_names, aucs, color=colors)
    axes[1, 0].set_xlabel('ROC-AUC Score', fontsize=10)
    axes[1, 0].set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    for i, v in enumerate(aucs):
        axes[1, 0].text(v + 0.02, i, f'{v:.3f}', va='center')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # ---- STEP 8: Chart 4 - Precision vs Recall ----
    axes[1, 1].plot(recalls, precisions, 'o-', markersize=10, linewidth=2, color='purple')
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name, (recalls[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Recall', fontsize=10)
    axes[1, 1].set_ylabel('Precision', fontsize=10)
    axes[1, 1].set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    # ---- STEP 9: Save figure ----
    plt.tight_layout()
    output_file = '../reports/model_comparison.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved chart to: {output_file}")
    
    # Show plot
    # plt.show()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
