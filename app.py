# app.py - WITH BASIC ML
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

# ========== PAGE CONFIG MUST BE FIRST ==========
st.set_page_config(
    page_title="Dropout Risk Advisor",
    page_icon="🎓",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# ========== APP HEADER ==========
st.title("🎓 Dropout Risk Advisor")
st.write("**Version:** ML Model Integrated")

# ========== LOAD DATA ==========
csv_path = "students_dropout_academic_success.csv"

@st.cache_data
def load_data():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    else:
        st.error("❌ CSV file not found!")
        st.stop()

with st.spinner("Loading data..."):
    df = load_data()
    st.success(f"✅ Loaded {len(df)} records")

# ========== BASIC ML MODEL ==========
@st.cache_resource
def train_simple_model():
    """Train a simple model for demo"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    if 'target' not in df.columns:
        st.error("❌ 'target' column not found in dataset")
        return None, None, None
    
    df_clean = df.copy()
    df_clean["y"] = df_clean["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})
    
    # Handle missing mappings
    df_clean = df_clean.dropna(subset=["y"])
    
    feature_cols = [c for c in df_clean.columns if c not in ["target", "y"]]
    X = df_clean[feature_cols]
    y = df_clean["y"]
    
    # Simple model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    return model, feature_cols, accuracy

# ========== TRAIN MODEL ==========
with st.spinner("Training ML model..."):
    model, feature_cols, accuracy = train_simple_model()
    
    if model:
        st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
        
        # Show feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Top 10 Important Features")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            importance_df = pd.DataFrame({
                'Feature': [feature_cols[i] for i in indices],
                'Importance': importances[indices]
            })
            st.dataframe(importance_df)

# ========== PREDICTION INTERFACE ==========
st.markdown("---")
st.subheader("🎯 Student Risk Assessment")

# Get important features (simplified list)
important_feats = [
    "Curricular units 2nd sem (approved)",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Admission grade",
    "Age at enrollment",
    "Gender"
]

# Filter to features that exist in data
available_feats = [f for f in important_feats if f in feature_cols] if feature_cols else important_feats

col1, col2 = st.columns(2)

user_input = {}
with col1:
    st.markdown("#### Academic Information")
    for feat in available_feats[:4]:  # First 4 features
        if feat in df.columns:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            median_val = float(df[feat].median())
            
            if feat == "Tuition fees up to date":
                val = st.selectbox(
                    feat,
                    options=[1, 0],
                    format_func=lambda x: "Up to date" if x == 1 else "Not up to date",
                    index=1
                )
            else:
                val = st.slider(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    step=0.1 if any(x in feat.lower() for x in ['grade', 'score']) else 1.0
                )
            user_input[feat] = val

with col2:
    st.markdown("#### Personal Information")
    for feat in available_feats[4:]:  # Last 3 features
        if feat in df.columns:
            if feat == "Gender":
                val = st.selectbox(
                    feat,
                    options=[1, 0],
                    format_func=lambda x: "Female" if x == 1 else "Male",
                    index=0
                )
            else:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                median_val = float(df[feat].median())
                
                val = st.slider(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    step=1.0
                )
            user_input[feat] = val

# ========== PREDICTION ==========
if st.button("🔍 Predict Dropout Risk", type="primary", use_container_width=True):
    if model and feature_cols:
        with st.spinner("Making prediction..."):
            # Prepare input
            medians = df[feature_cols].median()
            x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
            x_df = pd.DataFrame([x_full])
            
            # Predict
            probability = model.predict_proba(x_df)[0, 1]
            
            # Risk bands
            if probability < 0.2:
                risk = "Low"
                color = "green"
                action = "General support"
            elif probability < 0.5:
                risk = "Medium"
                color = "orange"
                action = "Progress monitoring"
            else:
                risk = "High"
                color = "red"
                action = "Intensive mentoring"
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric("Dropout Probability", f"{probability:.1%}")
            
            with col_result2:
                st.metric("Risk Level", risk)
            
            with col_result3:
                st.metric("Recommended Action", action)
            
            # Feature importance for this prediction
            st.subheader("🔍 Influencing Factors")
            
            # Get feature contributions (simplified)
            if hasattr(model, 'feature_importances_'):
                contributions = {}
                for i, feat in enumerate(feature_cols):
                    if feat in user_input:
                        # Simplified: use feature importance * normalized value
                        norm_val = (user_input[feat] - df[feat].mean()) / df[feat].std()
                        contributions[feat] = model.feature_importances_[i] * norm_val
                
                # Sort by absolute contribution
                sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                
                for feat, contrib in sorted_contrib:
                    direction = "increased" if contrib > 0 else "decreased"
                    st.write(f"• **{feat}** {direction} risk")
            
            st.success("✅ Prediction complete!")
    else:
        st.error("❌ Model not trained yet!")

# ========== DATASET INFO ==========
with st.expander("📊 Dataset Information", expanded=False):
    st.write(f"**Total records:** {len(df)}")
    st.write(f"**Features:** {len(df.columns)}")
    
    if 'target' in df.columns:
        st.write("**Target distribution:**")
        target_counts = df['target'].value_counts()
        st.bar_chart(target_counts)

st.success("🚀 Dropout Risk Advisor is fully functional!")
