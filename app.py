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
from sklearn.tree import plot_tree
import os
import sys

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Import models
from model.logistic import LogisticRegression
from model.decision_tree import DecisionTreeClassifier
from model.knn import KNNClassifier
from model.naive_bayes import NaiveBayesClassifier
from model.random_forest import RandomForestClassifier
from model.xgboost_model import XGBoostClassifier

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
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h4 {
        margin: 0;
        color: #666;
        font-size: 13px;
    }
    .metric-card h2 {
        margin: 4px 0 0 0;
        color: #1f77b4;
        font-size: 26px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        data_path = 'loan_data.csv'
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data"""
    # Separate features and target
    X = df.drop(['Target'], axis=1)
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

def train_model(model_name, X_train, y_train):
    """Train the selected model"""
    # Compute class imbalance ratio for models that need it
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    if model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        return model
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            criterion='entropy'
        )
        model.fit(X_train, y_train)
        return model
    elif model_name == "KNN":
        # Use SMOTE-style oversampling for KNN since it has no class_weight
        from imblearn.over_sampling import SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except Exception:
            X_res, y_res = X_train, y_train
        model = KNNClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            algorithm='auto'
        )
        model.fit(X_res, y_res)
        return model
    elif model_name == "Naive Bayes":
        # Use balanced class priors to handle imbalance
        model = NaiveBayesClassifier(
            var_smoothing=1e-7,
            priors=[0.5, 0.5]
        )
        model.fit(X_train, y_train)
        return model
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            criterion='entropy',
            max_features='sqrt'
        )
        model.fit(X_train, y_train)
        return model
    elif model_name == "XGBoost":
        model = XGBoostClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model
    else:
        return None

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'], ax=ax)
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
    # ── Header ──
    st.markdown('<h1 class="main-header">ML Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train, evaluate and compare classification models</p>', unsafe_allow_html=True)

    # ── Load training data ──
    df = load_data()
    if df is None:
        st.error("Failed to load `loan_data.csv`. Place it in the project root.")
        return

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)

    # ==================================================================
    # SIDEBAR
    # ==================================================================
    st.sidebar.title("⚙️ Settings")

    # Model selector
    model_options = ["Logistic Regression", "Decision Tree", "KNN",
                     "Naive Bayes", "Random Forest", "XGBoost"]
    selected_model = st.sidebar.selectbox("Model", model_options)

    # Threshold
    threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

    # ── Dataset section in sidebar ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Dataset")
    st.sidebar.caption(f"{df.shape[0]} rows  ·  {df.shape[1]} cols")

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "📥 Download Training Data",
        data=csv_bytes,
        file_name="loan_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Upload
    st.sidebar.markdown("")
    uploaded_file = st.sidebar.file_uploader("📤 Upload Test CSV", type=["csv"],
                                             help="CSV with same columns + Target")

    # ==================================================================
    # MAIN CONTENT — TABS
    # ==================================================================
    tab_single, tab_compare, tab_data = st.tabs(
        ["📈 Single Model", "🔄 Compare All Models", "📊 Dataset Info"]
    )

    # ------------------------------------------------------------------
    # TAB: Single Model
    # ------------------------------------------------------------------
    with tab_single:
        model = train_model(selected_model, X_train, y_train)
        if model is None:
            st.error(f"{selected_model} training failed.")
            return

        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= threshold).astype(int)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_proba >= threshold).astype(int)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_proba)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_test_pred)

        # ── Metrics row ──
        st.subheader(f"{selected_model} — Metrics")
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        for col, (label, val) in zip(
            [m1, m2, m3, m4, m5, m6, m7],
            [("Train Acc", train_acc), ("Test Acc", test_acc), ("AUC", auc),
             ("Precision", prec), ("Recall", rec), ("F1", f1), ("MCC", mcc)]
        ):
            col.metric(label, f"{val:.4f}")

        # ── Charts ──
        col_a, col_b = st.columns(2)

        cm = confusion_matrix(y_test, y_test_pred)
        with col_a:
            fig_cm = plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        with col_b:
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig_roc = plot_roc_curve(fpr, tpr, auc, selected_model)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        col_c, col_d = st.columns(2)
        with col_c:
            metrics_data = {
                'Accuracy': test_acc, 'AUC': auc, 'Precision': prec,
                'Recall': rec, 'F1': f1, 'MCC': (mcc + 1) / 2
            }
            fig_bar = plot_metrics_bar(metrics_data, selected_model)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with col_d:
            # Feature importance
            if hasattr(model, 'coef_'):
                fi = pd.DataFrame({'Feature': feature_names, 'Coef': model.coef_[0]})
                fi['Abs'] = np.abs(fi['Coef'])
                fi = fi.sort_values('Abs', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(7, 5))
                colors = ['green' if x > 0 else 'red' for x in fi['Coef']]
                ax.barh(range(len(fi)), fi['Coef'], color=colors)
                ax.set_yticks(range(len(fi)))
                ax.set_yticklabels(fi['Feature'], fontsize=9)
                ax.set_xlabel('Coefficient')
                ax.set_title('Top 10 Feature Importance', fontweight='bold')
                ax.axvline(x=0, color='black', lw=0.8)
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif hasattr(model, 'feature_importances_'):
                fi = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(range(len(fi)), fi['Importance'], color='green')
                ax.set_yticks(range(len(fi)))
                ax.set_yticklabels(fi['Feature'], fontsize=9)
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Feature importance not available for this model.")

        # Decision tree structure
        if selected_model == "Decision Tree":
            with st.expander("🌳 Decision Tree Structure (depth 3)"):
                fig_tree, ax = plt.subplots(figsize=(20, 12))
                plot_tree(model.model, feature_names=feature_names.tolist(),
                          class_names=['Not Approved', 'Approved'], filled=True,
                          rounded=True, fontsize=8, ax=ax, max_depth=3)
                plt.tight_layout()
                st.pyplot(fig_tree)
                plt.close(fig_tree)

        # Classification report
        with st.expander("📋 Classification Report"):
            report = classification_report(y_test, y_test_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB: Compare All Models
    # ------------------------------------------------------------------
    with tab_compare:
        # Decide evaluation data: uploaded CSV or default test split
        use_uploaded = False
        if uploaded_file is not None:
            data_source = st.radio(
                "Evaluate on",
                ["Default Test Split", "Uploaded CSV"],
                horizontal=True,
            )
            use_uploaded = data_source == "Uploaded CSV"

        if use_uploaded:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if 'Target' not in uploaded_df.columns:
                    st.error("Uploaded CSV must contain a **Target** column.")
                    use_uploaded = False
                else:
                    st.caption(f"Uploaded: {uploaded_file.name}  ·  {uploaded_df.shape[0]} rows")
                    with st.expander("Preview uploaded data"):
                        st.dataframe(uploaded_df.head(10), use_container_width=True)
                    drop_cols = [c for c in ['Target'] if c in uploaded_df.columns]
                    X_eval = uploaded_df.drop(drop_cols, axis=1)
                    y_eval = uploaded_df['Target']
                    cat_cols = X_eval.select_dtypes(include=['object']).columns.tolist()
                    for col in cat_cols:
                        X_eval[col] = LabelEncoder().fit_transform(X_eval[col].astype(str))
                    X_eval_scaled = scaler.transform(X_eval)
                    eval_label = "Uploaded Data"
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                use_uploaded = False

        if not use_uploaded:
            X_eval_scaled = X_test
            y_eval = y_test
            eval_label = "Test Split (20%)"

        st.caption(f"Evaluating on: **{eval_label}** — {len(y_eval):,} samples")

        all_names = ["Logistic Regression", "Decision Tree", "KNN",
                     "Naive Bayes", "Random Forest", "XGBoost"]
        results, roc_data = [], {}
        bar = st.progress(0, text="Evaluating models...")

        for i, name in enumerate(all_names):
            bar.progress(i / len(all_names), text=f"Training {name}...")
            m = train_model(name, X_train, y_train)
            if m is None:
                continue
            proba = m.predict_proba(X_eval_scaled)[:, 1]
            pred = (proba >= threshold).astype(int)
            acc = accuracy_score(y_eval, pred)
            try:
                auc_val = roc_auc_score(y_eval, proba)
            except ValueError:
                auc_val = 0.0
            results.append({
                'Model': name,
                'Accuracy': acc,
                'AUC': auc_val,
                'Precision': precision_score(y_eval, pred, zero_division=0),
                'Recall': recall_score(y_eval, pred, zero_division=0),
                'F1 Score': f1_score(y_eval, pred, zero_division=0),
                'MCC': matthews_corrcoef(y_eval, pred),
            })
            try:
                fp, tp, _ = roc_curve(y_eval, proba)
                roc_data[name] = (fp, tp, auc_val)
            except ValueError:
                pass

        bar.progress(1.0, text="Done!")
        res_df = pd.DataFrame(results)

        # ── Comparison table ──
        st.subheader("Metrics Comparison")

        fmt = {c: '{:.4f}' for c in ['Accuracy','AUC','Precision','Recall','F1 Score','MCC']}
        st.dataframe(res_df.style.format(fmt),
                     use_container_width=True, hide_index=True)

        best = res_df.loc[res_df['F1 Score'].idxmax()]
        st.success(f"🏆 **Best (F1): {best['Model']}** — "
                   f"Acc {best['Accuracy']:.4f}, AUC {best['AUC']:.4f}, "
                   f"F1 {best['F1 Score']:.4f}, MCC {best['MCC']:.4f}")

        # ── Charts side by side ──
        c1, c2 = st.columns(2)
        clrs = ['#2E86AB','#A23B72','#F18F01','#C73E1D',"#93B380",'#8E44AD']

        with c1:
            fig, ax = plt.subplots(figsize=(10, 5))
            mets = ['Accuracy','AUC','Precision','Recall','F1 Score']
            x = np.arange(len(mets))
            w = 0.12
            for i, (_, r) in enumerate(res_df.iterrows()):
                ax.bar(x + i*w, [r[m] for m in mets], w,
                       label=r['Model'], color=clrs[i%len(clrs)])
            ax.set_xticks(x + w*(len(res_df)-1)/2)
            ax.set_xticklabels(mets)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Score')
            ax.set_title('Metrics Comparison', fontweight='bold')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, (nm, (fp, tp, au)) in enumerate(roc_data.items()):
                ax.plot(fp, tp, lw=2, color=clrs[i%len(clrs)],
                        label=f'{nm} ({au:.3f})')
            ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curves', fontweight='bold')
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Confusion matrices ──
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, name in enumerate(all_names):
            m = train_model(name, X_train, y_train)
            if m is None:
                continue
            proba = m.predict_proba(X_eval_scaled)[:, 1]
            pred = (proba >= threshold).astype(int)
            cm_m = confusion_matrix(y_eval, pred)
            with cols[i % 3]:
                fig_cm = plot_confusion_matrix(cm_m, name)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

        # README table
        with st.expander("📝 README Table"):
            lines = "| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |\n"
            lines += "|-------|----------|-----|-----------|--------|----------|-----|\n"
            for _, r in res_df.iterrows():
                lines += (f"| {r['Model']} | {r['Accuracy']:.4f} | {r['AUC']:.4f} | "
                          f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                          f"{r['F1 Score']:.4f} | {r['MCC']:.4f} |\n")
            st.code(lines, language="markdown")

    # ------------------------------------------------------------------
    # TAB: Dataset Info
    # ------------------------------------------------------------------
    with tab_data:
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", f"{df.shape[0]:,}")
        c2.metric("Features", f"{df.shape[1]}")
        c3.metric("Classes", f"{df['Target'].nunique()}")

        st.dataframe(df.head(10), use_container_width=True)

        target_counts = df['Target'].value_counts()
        st.bar_chart(target_counts)

        class_0 = target_counts.get(0, 0)
        class_1 = target_counts.get(1, 0)
        ratio = class_0 / class_1 if class_1 > 0 else 0
        st.info(
            f"**Not Approved (0):** {class_0:,} ({class_0/len(df)*100:.1f}%)  ·  "
            f"**Approved (1):** {class_1:,} ({class_1/len(df)*100:.1f}%)  ·  "
            f"**Ratio:** {ratio:.1f}:1"
        )

    # Footer
    st.markdown("---")
    st.caption("Machine Learning Assignment 2  ·  Built with Streamlit")

if __name__ == "__main__":
    main()
