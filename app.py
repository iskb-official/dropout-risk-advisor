# app.py - Streamlit Cloud Ready Version
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import base64
from io import StringIO

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# ========== CONFIGURATION ==========
CSV_NAME = "students_dropout_academic_success.csv"
LOW_TH = 0.2
HIGH_TH = 0.5

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #D1FAE5 !important;
        border-left-color: #10B981 !important;
        color: #065F46;
    }
    .risk-medium {
        background-color: #FEF3C7 !important;
        border-left-color: #F59E0B !important;
        color: #92400E;
    }
    .risk-high {
        background-color: #FEE2E2 !important;
        border-left-color: #EF4444 !important;
        color: #991B1B;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== HELPER FUNCTIONS ==========
def assign_risk_band(p: float) -> str:
    """Assign risk band based on probability"""
    if p < LOW_TH:
        return "low"
    elif p < HIGH_TH:
        return "medium"
    else:
        return "high"

def create_risk_gauge(p_dropout):
    """Create HTML/CSS risk gauge"""
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0;'>
        <div style='position: relative; height: 120px; width: 300px; margin: 0 auto;'>
            <div style='position: absolute; width: 100%; height: 20px; 
                 background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%); 
                 border-radius: 10px;'></div>
            <div style='position: absolute; left: {p_dropout*100}%; top: -10px; 
                 width: 2px; height: 40px; background-color: #000; transform: translateX(-50%);'></div>
            <div style='position: absolute; left: {p_dropout*100}%; top: 30px; 
                 transform: translateX(-50%); font-weight: bold; font-size: 1.2rem;'>
                {p_dropout:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== CACHE FUNCTIONS ==========
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv(CSV_NAME)
        st.success(f"✅ Successfully loaded dataset with {len(df)} records")
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{CSV_NAME}' not found! Please make sure it's in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def train_model():
    """Train and cache the ML model"""
    try:
        df = load_data()
        
        # Check if target column exists
        if 'target' not in df.columns:
            st.error("❌ Dataset must contain 'target' column")
            st.stop()
        
        # Map target to binary
        df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})
        
        # Handle missing values if any
        if df["y"].isnull().any():
            st.warning("⚠️ Some target values couldn't be mapped. Dropping those rows.")
            df = df.dropna(subset=["y"])
        
        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols]
        y = df["y"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Lightweight model for cloud
        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,  # Important for Streamlit Cloud
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        return model, explainer, feature_cols, X_train
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        st.stop()

# ========== MAIN APP ==========
def main():
    st.set_page_config(
        page_title="Dropout Risk Advisor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<h1 style='text-align: center;'>🎓</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 class='main-header'>Trustworthy Dropout Risk Advisor</h1>", unsafe_allow_html=True)
    
    # Load model with progress
    with st.spinner("🤖 Loading AI model and data..."):
        model, explainer, feature_cols, X_train = train_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dataset Info")
        df = load_data()
        st.write(f"**Total students:** {len(df):,}")
        st.write(f"**Features:** {len(feature_cols)}")
        
        # Target distribution
        st.markdown("### 🎯 Target Distribution")
        target_counts = df['target'].value_counts()
        for target, count in target_counts.items():
            st.write(f"**{target}:** {count} ({count/len(df)*100:.1f}%)")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool predicts dropout risk using machine learning.
        
        **Model:** XGBoost Classifier
        **Features:** 35 academic & demographic factors
        **Accuracy:** ~87% (test set)
        """)
    
    # Main content
    st.markdown("### 🎯 Student Risk Assessment")
    
    # Two-column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📝 Enter Student Information")
        
        # Important features (you can customize this list)
        important_feats = [
            "Curricular units 2nd sem (approved)",
            "Tuition fees up to date", 
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Admission grade",
            "Age at enrollment",
            "Gender"
        ]
        
        user_input = {}
        for feat in important_feats:
            if feat in ["Tuition fees up to date", "Gender"]:
                if feat == "Tuition fees up to date":
                    val = st.selectbox(
                        feat,
                        options=[1, 0],
                        format_func=lambda x: "Up to date" if x == 1 else "Not up to date",
                        help="Whether tuition fees are paid on time"
                    )
                else:  # Gender
                    val = st.selectbox(
                        feat,
                        options=[1, 0],
                        format_func=lambda x: "Female" if x == 1 else "Male",
                        help="Student's gender"
                    )
            else:
                # Get min/max from training data
                col_data = X_train[feat]
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                median_val = float(col_data.median())
                
                val = st.slider(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    help=f"Range in training data: {min_val:.1f} to {max_val:.1f}"
                )
            user_input[feat] = val
    
    with col_right:
        st.markdown("#### 📊 Quick Stats")
        
        # Model info
        st.markdown("""
        <div class='metric-box'>
        <h3>Model Ready</h3>
        <p>Trained on student data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Risk Bands")
        st.info("""
        **Low Risk:** < 20%  
        **Medium Risk:** 20-50%  
        **High Risk:** ≥ 50%
        """)
    
    # Evaluate button
    st.markdown("---")
    if st.button("🔍 Evaluate Dropout Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing student profile..."):
            time.sleep(0.5)
            
            # Prepare full feature vector
            medians = X_train.median()
            x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
            x_df = pd.DataFrame([x_full])
            
            # Make prediction
            p_dropout = float(model.predict_proba(x_df)[:, 1])
            band = assign_risk_band(p_dropout)
            
            # Determine intervention
            interventions = {
                "high": "🎯 **Intensive mentoring and academic counseling**",
                "medium": "📈 **Skills workshop and progress monitoring**", 
                "low": "✅ **Regular check-ins and general support**"
            }
            action = interventions[band]
            
            # Calculate SHAP values
            shap_vals = explainer.shap_values(x_df)
            shap_row = shap_vals[0]
            
            # Display results
            st.markdown("## 📋 Assessment Results")
            
            # Results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Risk Probability")
                create_risk_gauge(p_dropout)
            
            with col2:
                risk_class = f"risk-{band}"
                st.markdown(f"""
                <div class='card {risk_class}'>
                <h3>Risk Classification</h3>
                <h1 style='font-size: 3rem; margin: 10px 0;'>{band.upper()}</h1>
                <p><strong>Probability:</strong> {p_dropout:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='card'>
                <h3>Recommended Intervention</h3>
                <p style='font-size: 1.1rem;'>{action}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # SHAP analysis
            st.markdown("### 🔍 Key Influencing Factors")
            
            # Get top 5 factors
            idx = np.argsort(-np.abs(shap_row))[:5]
            
            col_shap1, col_shap2 = st.columns([3, 2])
            
            with col_shap1:
                # Create SHAP visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                features = [feature_cols[i] for i in idx]
                values = shap_row[idx]
                colors = ['#EF4444' if v > 0 else '#10B981' for v in values]
                
                y_pos = np.arange(len(features))
                ax.barh(y_pos, values, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Impact on Dropout Risk (SHAP value)')
                ax.set_title('Top Factors Affecting Prediction')
                
                # Add value labels
                for i, v in enumerate(values):
                    ax.text(v, i, f' {v:+.3f}', va='center', 
                           color='white' if abs(v) > 0.05 else 'black')
                
                st.pyplot(fig)
            
            with col_shap2:
                st.markdown("#### 📝 Explanation")
                
                explanations = []
                for i in idx:
                    feat = feature_cols[i]
                    val = shap_row[i]
                    direction = "increases" if val > 0 else "decreases"
                    explanations.append(f"**{feat}** {direction} risk")
                
                st.markdown(f"""
                <div class='card'>
                <p>The model predicts <strong>{p_dropout:.1%}</strong> dropout probability.</p>
                <p><strong>Main reasons:</strong></p>
                <ul>
                {''.join([f'<li>{exp}</li>' for exp in explanations[:3]])}
                </ul>
                <p><strong>Action:</strong> {action}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download results
            st.markdown("---")
            results_text = f"""Dropout Risk Assessment
========================

Probability: {p_dropout:.1%}
Risk Level: {band.upper()}
Intervention: {action}

Top Factors:
{chr(10).join([f'- {feature_cols[i]}: {shap_row[i]:+.3f}' for i in idx])}

Assessment Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
            
            st.download_button(
                label="📥 Download Assessment Report",
                data=results_text,
                file_name=f"dropout_risk_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()