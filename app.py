# Streamlit Web Application
# Machine Learning Assignment 2 - Classification Models

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import sys

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Import models
from model.logistic import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        data_path = 'clean_data.csv'
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data"""
    # Separate features and target
    X = df.drop(['ID', 'Target'], axis=1)
    y = df['Target']
    
    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Train the selected model"""
    if model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        return model
    else:
        return None

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'], ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    return fig

def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig

def plot_metrics_bar(metrics_data, title="Performance Metrics"):
    """Plot metrics bar chart"""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), 
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E44AD'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Models</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive model evaluation and prediction interface")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_options = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # Add threshold slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Parameters")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Increase threshold to reduce false positives (but may increase false negatives)"
    )
    
    st.sidebar.info(f"""
    **Current Threshold: {threshold}**
    - **Lower** → More positive predictions (↑ FP, ↑ TP)
    - **Higher** → Fewer positive predictions (↓ FP, ↓ TP)
    """)
    
    # Available models
    implemented_models = ["Logistic Regression"]
    
    if selected_model not in implemented_models:
        st.warning(f"⚠️ {selected_model} is not yet implemented. Please select 'Logistic Regression'.")
        return
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please ensure 'clean_data.csv' exists in the project directory.")
        return
    
    # Display dataset info
    st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Show dataset option
    if st.sidebar.checkbox("Show Dataset Info"):
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        overview_metrics = [
            ("Total Samples", df.shape[0]),
            ("Total Features", df.shape[1]),
            ("Target Classes", df['Target'].nunique())
        ]
        
        cols = [col1, col2, col3]
        for i, (name, value) in enumerate(overview_metrics):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h4 style='margin: 0; color: #666; font-size: 14px;'>{name}</h4>
                    <h2 style='margin: 5px 0 0 0; color: #1f77b4; font-size: 32px;'>{value:,}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write("**Target Distribution:**")
        target_counts = df['Target'].value_counts()
        st.bar_chart(target_counts)
        
        # Show class imbalance ratio
        class_0 = target_counts[0]
        class_1 = target_counts[1] if 1 in target_counts.index else 0
        imbalance_ratio = class_0 / class_1 if class_1 > 0 else 0
        
        st.warning(f"""
        **⚠️ Class Imbalance Detected!**
        - Class 0: {class_0:,} samples ({class_0/df.shape[0]*100:.1f}%)
        - Class 1: {class_1:,} samples ({class_1/df.shape[0]*100:.1f}%)
        - Imbalance Ratio: {imbalance_ratio:.1f}:1
        
        **This is why you see high False Positives:**
        - The model uses `class_weight='balanced'` to compensate for imbalance
        - This makes the model more aggressive in predicting Class 1
        - Result: More True Positives (good) but also More False Positives
        - **Solution**: Adjust the threshold slider in the sidebar (try 0.6-0.7)
        """)
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)
    
    st.sidebar.success(f"✅ Data preprocessed")
    
    # Train model
    with st.spinner(f"Training {selected_model}..."):
        model = train_model(selected_model, X_train, y_train)
    
    if model is None:
        st.error(f"Model {selected_model} training failed.")
        return
    
    st.sidebar.success(f"✅ {selected_model} trained successfully")
    
    # Make predictions with custom threshold
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    
    # Display metrics
    st.header(f"📈 {selected_model} - Performance Metrics")
    
    # Create metrics DataFrame for better display
    metrics_df = pd.DataFrame({
        'Metric': ['Training Accuracy', 'Test Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
        'Value': [
            f"{train_accuracy:.4f}",
            f"{test_accuracy:.4f}",
            f"{test_auc:.4f}",
            f"{test_precision:.4f}",
            f"{test_recall:.4f}",
            f"{test_f1:.4f}",
            f"{test_mcc:.4f}"
        ]
    })
    
    # Display as columns with styled boxes
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_list = [
        ("Training Accuracy", train_accuracy),
        ("Test Accuracy", test_accuracy),
        ("AUC Score", test_auc),
        ("Precision", test_precision),
        ("Recall", test_recall),
        ("F1 Score", test_f1),
        ("MCC", test_mcc)
    ]
    
    cols = [col1, col2, col3, col4, col1, col2, col3]
    
    for i, (metric_name, metric_value) in enumerate(metrics_list):
        with cols[i]:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;'>
                <h4 style='margin: 0; color: #666; font-size: 14px;'>{metric_name}</h4>
                <h2 style='margin: 5px 0 0 0; color: #1f77b4; font-size: 28px;'>{metric_value:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.header("📊 Visualizations")
    
    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)
    
    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        fig_cm = plot_confusion_matrix(cm, f"{selected_model}")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    
    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        fig_roc = plot_roc_curve(fpr, tpr, test_auc, f"{selected_model}")
        st.pyplot(fig_roc)
        plt.close(fig_roc)
    
    # Confusion Matrix Breakdown
    st.markdown("### Confusion Matrix Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    
    cm_metrics = [
        ("True Negatives", int(cm[0][0]), "✅"),
        ("False Positives", int(cm[0][1]), "❌"),
        ("False Negatives", int(cm[1][0]), "❌"),
        ("True Positives", int(cm[1][1]), "✅")
    ]
    
    cols = [col1, col2, col3, col4]
    for i, (name, value, emoji) in enumerate(cm_metrics):
        with cols[i]:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;'>
                <h4 style='margin: 0; color: #666; font-size: 14px;'>{emoji} {name}</h4>
                <h2 style='margin: 5px 0 0 0; color: #1f77b4; font-size: 28px;'>{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Metrics Summary and Feature Importance in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrics Summary")
        metrics_data = {
            'Accuracy': test_accuracy,
            'AUC': test_auc,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1,
            'MCC': (test_mcc + 1) / 2
        }
        fig_metrics = plot_metrics_bar(metrics_data, f"{selected_model}")
        st.pyplot(fig_metrics)
        plt.close(fig_metrics)
    
    with col2:
        st.subheader("Feature Importance")
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0]
            })
            feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
            feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
            ax.barh(range(len(feature_importance)), feature_importance['Coefficient'], color=colors)
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['Feature'], fontsize=9)
            ax.set_xlabel('Coefficient Value', fontsize=10)
            ax.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature importance not available for this model.")
    
    # Classification Report
    st.header("📋 Detailed Classification Report")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Model summary for README
    st.header("📝 Summary for README")
    summary_text = f"| {selected_model} | {test_accuracy:.4f} | {test_auc:.4f} | {test_precision:.4f} | {test_recall:.4f} | {test_f1:.4f} | {test_mcc:.4f} |"
    st.code(summary_text, language="markdown")
    
    # Footer
    st.markdown("---")
    st.markdown("**Machine Learning Assignment 2** | Built with Streamlit 🚀")

if __name__ == "__main__":
    main()
