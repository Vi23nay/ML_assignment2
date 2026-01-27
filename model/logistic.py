# Logistic Regression Implementation using Sklearn

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegression:
    """
    Logistic Regression classifier using sklearn library.
    
    Parameters:
    -----------
    max_iter : int, default=5000
        Maximum number of iterations for solver to converge
    class_weight : str or dict, default='balanced'
        Weights associated with classes
    random_state : int, default=42
        Random state for reproducibility
    solver : str, default='lbfgs'
        Algorithm to use in optimization problem
    """
    
    def __init__(self, max_iter=5000, class_weight='balanced', random_state=42, solver='lbfgs'):
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.model = SklearnLogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver
        )
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the logistic regression model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
            
        Returns:
        --------
        self
            Fitted estimator
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
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
            Predicted class labels (0 or 1)
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
# Training and Evaluation on clean_data.csv
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

    # Load the dataset (adjust path based on current directory)
    print("Loading dataset...")
    data_path = '../clean_data.csv' if os.path.exists('../clean_data.csv') else 'clean_data.csv'
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nTarget distribution:")
    print(df['Target'].value_counts())

    # Separate features and target
    X = df.drop(['ID', 'Target'], axis=1)
    y = df['Target']

    print(f"\nFeature columns: {X.columns.tolist()}")

    # Check for categorical columns that need encoding
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns: {categorical_columns}")

    # Encode categorical variables if any
    if categorical_columns:
        print("Encoding categorical variables...")
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Split the data into training and testing sets
    print("\nSplitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # Feature Scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model using sklearn
    print("\n" + "="*60)
    print("Training Logistic Regression Model (sklearn)...")
    print("="*60)

    model = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
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
    print("\nLogistic Regression Results:")
    print(f"| Logistic Regression | {test_accuracy:.4f} | {test_auc:.4f} | {test_precision:.4f} | {test_recall:.4f} | {test_f1:.4f} | {test_mcc:.4f} |")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)

    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logistic Regression - Model Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)

    # Add text annotations for better clarity
    axes[0, 0].text(0.5, -0.15, f'TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}', 
                    ha='center', transform=axes[0, 0].transAxes, fontsize=10)

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Model Information
    model_info = f"""Model: Logistic Regression (sklearn)
Solver: {model.solver}
Max Iterations: {model.max_iter}
Class Weight: {model.class_weight}
Number of Features: {len(model.coef_[0])}
Intercept: {model.intercept_[0]:.4f}

Training Samples: {len(X_train_scaled)}
Test Samples: {len(X_test_scaled)}
"""
    axes[1, 0].text(0.1, 0.9, model_info, fontsize=12, family='monospace',
                    verticalalignment='top', transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Model Information', fontsize=14, fontweight='bold')

    # 4. Feature Importance (Top 10 features by coefficient magnitude)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(10)
    
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
    axes[1, 1].barh(range(len(feature_importance)), feature_importance['Coefficient'], color=colors)
    axes[1, 1].set_yticks(range(len(feature_importance)))
    axes[1, 1].set_yticklabels(feature_importance['Feature'])
    axes[1, 1].set_xlabel('Coefficient Value', fontsize=12)
    axes[1, 1].set_title('Top 10 Feature Importance (by Coefficient)', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    
    # Save the figure
    os.makedirs('../visualizations', exist_ok=True)
    plt.savefig('../visualizations/logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: visualizations/logistic_regression_analysis.png")
    
    # Show the plot
    plt.show()

    # ========================================================================
    # Additional: Metrics Comparison Bar Chart
    # ========================================================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    metrics_data = {
        'Accuracy': test_accuracy,
        'AUC': test_auc,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'MCC': (test_mcc + 1) / 2  # Normalize MCC from [-1,1] to [0,1] for visualization
    }
    
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E44AD'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Logistic Regression - Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/logistic_regression_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved to: visualizations/logistic_regression_metrics.png")
    plt.show()

    print("\n" + "="*60)
    print("✓ Training and Evaluation Complete!")
    print("="*60)
