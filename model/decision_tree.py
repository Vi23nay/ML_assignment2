# Decision Tree Classifier Implementation using Sklearn

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

class DecisionTreeClassifier:
    """
    Decision Tree classifier using sklearn library.
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    class_weight : str or dict, default='balanced'
        Weights associated with classes
    random_state : int, default=42
        Random state for reproducibility
    criterion : str, default='gini'
        Function to measure the quality of a split ('gini' or 'entropy')
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 class_weight='balanced', random_state=42, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.criterion = criterion
        self.model = SklearnDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            criterion=criterion
        )
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """
        Fit the decision tree model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self
            Fitted estimator
        """
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        array-like, shape (n_samples, 2)
            Predicted probabilities for each class
        """
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        return self.model.score(X, y)


# ============================================================================
# Training and Evaluation on loan_data.csv
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score, 
        roc_auc_score, 
        precision_score, 
        recall_score, 
        f1_score,
        matthews_corrcoef,
        confusion_matrix,
        classification_report,
        roc_curve
    )
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import plot_tree

    # Load the dataset
    print("Loading dataset...")
    data_path = '../loan_data.csv' if os.path.exists('../loan_data.csv') else 'loan_data.csv'
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nTarget distribution:")
    print(df['Target'].value_counts())

    # Separate features and target
    X = df.drop(['Target'], axis=1)
    y = df['Target']

    print(f"\nFeature columns: {X.columns.tolist()}")

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns: {categorical_columns}")

    if categorical_columns:
        print("Encoding categorical variables...")
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Split the data
    print("\nSplitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # Feature Scaling (optional for Decision Trees but keeping for consistency)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Decision Tree model
    print("\n" + "="*60)
    print("Training Decision Tree Model (sklearn)...")
    print("="*60)

    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        criterion='gini'
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")

    # Testing metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    print("\n--- Test Set Performance ---")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"AUC Score: {test_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"MCC:       {test_mcc:.4f}")

    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_test_pred))

    # Summary for README
    print("\n" + "="*60)
    print("SUMMARY FOR README")
    print("="*60)
    print("\nDecision Tree Results:")
    print(f"| Decision Tree | {test_accuracy:.4f} | {test_auc:.4f} | {test_precision:.4f} | {test_recall:.4f} | {test_f1:.4f} | {test_mcc:.4f} |")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Decision Tree - Model Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True, 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].text(0.5, -0.15, f'TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}', 
                    ha='center', transform=axes[0, 0].transAxes, fontsize=10)

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Model Information
    model_info = f"""Model: Decision Tree (sklearn)
Criterion: {model.criterion}
Max Depth: {model.max_depth}
Min Samples Split: {model.min_samples_split}
Min Samples Leaf: {model.min_samples_leaf}
Class Weight: {model.class_weight}

Training Samples: {len(X_train_scaled)}
Test Samples: {len(X_test_scaled)}
"""
    axes[1, 0].text(0.1, 0.9, model_info, fontsize=12, family='monospace',
                    verticalalignment='top', transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Model Information', fontsize=14, fontweight='bold')

    # 4. Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
    
    axes[1, 1].barh(range(len(feature_importance)), feature_importance['Importance'], color='green')
    axes[1, 1].set_yticks(range(len(feature_importance)))
    axes[1, 1].set_yticklabels(feature_importance['Feature'])
    axes[1, 1].set_xlabel('Importance', fontsize=12)
    axes[1, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    
    os.makedirs('../visualizations', exist_ok=True)
    plt.savefig('../visualizations/decision_tree_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: visualizations/decision_tree_analysis.png")
    plt.show()

    # Metrics Bar Chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    metrics_data = {
        'Accuracy': test_accuracy,
        'AUC': test_auc,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'MCC': (test_mcc + 1) / 2
    }
    
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), color='green')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Decision Tree - Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/decision_tree_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved to: visualizations/decision_tree_metrics.png")
    plt.show()

    # ========================================================================
    # Decision Tree Visualization
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING DECISION TREE STRUCTURE...")
    print("="*60)
    
    fig3, ax = plt.subplots(figsize=(25, 15))
    plot_tree(model.model, 
              feature_names=X.columns.tolist(),
              class_names=['Class 0', 'Class 1'],
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax,
              max_depth=3)  # Limit depth for readability
    
    plt.title('Decision Tree Structure (Max Depth 3 shown)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../visualizations/decision_tree_structure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Tree structure saved to: visualizations/decision_tree_structure.png")
    plt.show()

    print("\n" + "="*60)
    print("✓ Training and Evaluation Complete!")
    print("="*60)
