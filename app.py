# app.py - COMPLETE DROPOUT RISK ADVISOR
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG MUST BE FIRST ==========
st.set_page_config(
    page_title="Dropout Risk Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .sub-header {
        font-size: 1.5rem !important;
        color: #374151 !important;
        font-weight: 600;
        margin-top: 1.5rem;
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
    .feature-box {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 20px;
        background-color: #E5E7EB;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== CONFIGURATION ==========
CSV_NAME = "students_dropout_academic_success.csv"
LOW_TH = 0.2
HIGH_TH = 0.5

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
    """Create HTML/CSS risk gauge visualization"""
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0;'>
        <div style='position: relative; height: 120px; width: 100%; max-width: 400px; margin: 0 auto;'>
            <div style='position: absolute; width: 100%; height: 20px; 
                 background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%); 
                 border-radius: 10px;'></div>
            <div style='position: absolute; left: {min(p_dropout*100, 100)}%; top: -10px; 
                 width: 2px; height: 40px; background-color: #000; transform: translateX(-50%);'></div>
            <div style='position: absolute; left: {min(p_dropout*100, 100)}%; top: 30px; 
                 transform: translateX(-50%); font-weight: bold; font-size: 1.2rem;'>
                {p_dropout:.1%}
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 10px; width: 100%; max-width: 400px; margin: 10px auto;'>
            <span>0%</span>
            <span style='font-weight: bold;'>Dropout Risk</span>
            <span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        if not os.path.exists(CSV_NAME):
            st.error(f"❌ CSV file '{CSV_NAME}' not found!")
            st.info("💡 Make sure the CSV file is uploaded to GitHub repository")
            return None
        
        df = pd.read_csv(CSV_NAME)
        st.success(f"✅ Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# ========== MODEL TRAINING ==========
@st.cache_resource
def train_model(_df):
    """Train XGBoost model with error handling"""
    try:
        # Check if required packages are installed
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split
            import shap
        except ImportError as e:
            st.error(f"❌ Missing package: {e}")
            st.info("💡 Update requirements.txt with: xgboost==1.7.6, scikit-learn==1.3.0, shap==0.41.0")
            return None, None, None, None
        
        # Prepare data
        df = _df.copy()
        
        if 'target' not in df.columns:
            st.error("❌ Dataset must contain 'target' column")
            return None, None, None, None
        
        # Map target to binary
        df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})
        
        # Handle missing values
        if df["y"].isnull().any():
            df = df.dropna(subset=["y"])
        
        if len(df) == 0:
            st.error("❌ No valid data after cleaning")
            return None, None, None, None
        
        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols]
        y = df["y"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=42,
            n_jobs=1,  # Important for Streamlit Cloud
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        return model, explainer, feature_cols, accuracy
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None, None, None

# ========== MAIN APP ==========
def main():
    # App header
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("<h1 style='text-align: center;'>🎓</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<h1 class='main-header'>Trustworthy Dropout Risk Advisor</h1>", unsafe_allow_html=True)
    
    st.write("This tool predicts student dropout risk using machine learning and provides intervention recommendations.")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 📊 Dataset Info")
        
        # Load data
        with st.spinner("Loading data..."):
            df = load_data()
        
        if df is not None:
            st.write(f"**Total students:** {len(df):,}")
            st.write(f"**Features:** {len(df.columns)}")
            
            if 'target' in df.columns:
                st.markdown("### 🎯 Target Distribution")
                target_counts = df['target'].value_counts()
                for target, count in target_counts.items():
                    percentage = count / len(df) * 100
                    st.write(f"**{target}:** {count} ({percentage:.1f}%)")
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        st.info("""
        **Algorithm:** XGBoost Classifier
        **Training:** 80% of data
        **Validation:** 20% of data
        **Risk Thresholds:**
        - Low: < 20%
        - Medium: 20-50%
        - High: ≥ 50%
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool helps identify students at risk of dropping out and recommends appropriate interventions.
        
        **Features:**
        - ML-based dropout prediction
        - SHAP-based explanations
        - Fairness-aware analysis
        - Actionable recommendations
        """)
    
    # ========== MAIN CONTENT ==========
    if df is None:
        st.error("❌ Cannot proceed without data. Please check CSV file.")
        return
    
    # Train model
    st.markdown("### 🤖 Machine Learning Model")
    
    with st.spinner("Training AI model (this may take a moment)..."):
        model, explainer, feature_cols, accuracy = train_model(df)
    
    if model is None:
        st.error("❌ Model training failed. Check requirements.txt and try again.")
        st.info("""
        **Required packages:**
        - xgboost==1.7.6
        - scikit-learn==1.3.0
        - shap==0.41.0
        - pandas==1.5.3
        - numpy==1.24.3
        """)
        return
    
    st.success(f"✅ Model trained successfully! Test accuracy: {accuracy:.2%}")
    
    # ========== STUDENT ASSESSMENT ==========
    st.markdown("---")
    st.markdown("## 🎯 Student Risk Assessment")
    
    # Select important features (simplified for UI)
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
    available_feats = [f for f in important_feats if f in df.columns]
    
    if not available_feats:
        st.warning("⚠️ Important features not found in dataset. Using all features.")
        available_feats = feature_cols[:7]  # Use first 7 features
    
    # Two-column input form
    col_left, col_right = st.columns(2)
    
    user_input = {}
    with col_left:
        st.markdown("#### 📝 Academic Information")
        for feat in available_feats[:4]:  # First 4 features
            if feat in df.columns:
                col_data = df[feat]
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                median_val = float(col_data.median())
                
                if feat == "Tuition fees up to date":
                    val = st.selectbox(
                        feat,
                        options=[1, 0],
                        format_func=lambda x: "Up to date" if x == 1 else "Not up to date",
                        help="Whether tuition fees are paid on time"
                    )
                elif "grade" in feat.lower() or "score" in feat.lower():
                    val = st.slider(
                        feat,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=0.5,
                        help=f"Range: {min_val:.1f} to {max_val:.1f}"
                    )
                else:
                    val = st.slider(
                        feat,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=1.0,
                        help=f"Range: {min_val:.0f} to {max_val:.0f}"
                    )
                user_input[feat] = val
    
    with col_right:
        st.markdown("#### 👤 Personal Information")
        for feat in available_feats[4:]:  # Remaining features
            if feat in df.columns:
                if feat == "Gender":
                    val = st.selectbox(
                        feat,
                        options=[1, 0],
                        format_func=lambda x: "Female" if x == 1 else "Male",
                        help="Student's gender"
                    )
                else:
                    col_data = df[feat]
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    median_val = float(col_data.median())
                    
                    val = st.slider(
                        feat,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=1.0,
                        help=f"Range: {min_val:.0f} to {max_val:.0f}"
                    )
                user_input[feat] = val
    
    # ========== PREDICTION BUTTON ==========
    st.markdown("---")
    predict_button = st.button(
        "🔍 Predict Dropout Risk",
        type="primary",
        use_container_width=True,
        help="Click to analyze student's dropout risk"
    )
    
    # ========== PREDICTION RESULTS ==========
    if predict_button:
        with st.spinner("Analyzing student profile..."):
            time.sleep(0.5)  # Small delay for UX
            
            # Prepare input data
            medians = df[feature_cols].median()
            x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
            x_df = pd.DataFrame([x_full])
            
            # Make prediction
            try:
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
                
                # ========== DISPLAY RESULTS ==========
                st.markdown("## 📋 Assessment Results")
                
                # Results in columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown("#### Risk Probability")
                    create_risk_gauge(p_dropout)
                
                with res_col2:
                    risk_class = f"risk-{band}"
                    st.markdown(f"""
                    <div class='card {risk_class}'>
                    <h3>Risk Classification</h3>
                    <h1 style='font-size: 3rem; margin: 10px 0;'>{band.upper()}</h1>
                    <p><strong>Probability:</strong> {p_dropout:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with res_col3:
                    st.markdown(f"""
                    <div class='card'>
                    <h3>Recommended Intervention</h3>
                    <p style='font-size: 1.1rem;'>{action}</p>
                    <p><small>Based on institutional policy for {band} risk students</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ========== SHAP ANALYSIS ==========
                st.markdown("## 🔍 Key Influencing Factors")
                
                # Get top 5 factors
                idx = np.argsort(-np.abs(shap_row))[:5]
                
                col_shap1, col_shap2 = st.columns([3, 2])
                
                with col_shap1:
                    st.markdown("#### 📊 Impact Visualization")
                    
                    # Simple bar chart using Streamlit
                    top_features = [feature_cols[i] for i in idx]
                    top_values = [float(shap_row[i]) for i in idx]
                    
                    # Create color-coded bars
                    for feat, val in zip(top_features, top_values):
                        color = "#EF4444" if val > 0 else "#10B981"
                        direction = "increased" if val > 0 else "decreased"
                        width = min(abs(val) * 20, 100)  # Scale for visualization
                        
                        st.markdown(f"""
                        <div style='margin: 10px 0;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                                <span><strong>{feat}</strong></span>
                                <span style='color: {color};'>{val:+.3f}</span>
                            </div>
                            <div style='height: 10px; background-color: #E5E7EB; border-radius: 5px; overflow: hidden;'>
                                <div style='height: 100%; width: {width}%; background-color: {color}; border-radius: 5px;'></div>
                            </div>
                            <div style='font-size: 0.8rem; color: #666; margin-top: 2px;'>
                                This factor {direction} the dropout risk
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_shap2:
                    st.markdown("#### 📝 Explanation")
                    
                    explanations = []
                    for i in idx[:3]:  # Top 3 factors
                        feat = feature_cols[i]
                        val = shap_row[i]
                        direction = "increased" if val > 0 else "decreased"
                        magnitude = "significantly" if abs(val) > 0.1 else "slightly"
                        explanations.append(f"**{feat}** {magnitude} {direction} the risk")
                    
                    st.markdown(f"""
                    <div class='info-box'>
                    <p>The model predicts a <strong>{p_dropout:.1%} probability</strong> of dropout for this student.</p>
                    <p><strong>Main reasons:</strong></p>
                    <ul>
                    {''.join([f'<li>{exp}</li>' for exp in explanations])}
                    </ul>
                    <p><strong>Action needed:</strong> {action.split('**')[1].split('**')[0]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ========== DOWNLOAD REPORT ==========
                st.markdown("---")
                st.markdown("### 📥 Download Report")
                
                results_text = f"""Dropout Risk Assessment Report
================================

Student Assessment Details:
- Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
- Model Accuracy: {accuracy:.2%}

Prediction Results:
- Dropout Probability: {p_dropout:.1%}
- Risk Level: {band.upper()}
- Recommended Intervention: {action.split('**')[1].split('**')[0]}

Top Influencing Factors:
{chr(10).join([f'- {feature_cols[i]}: {shap_row[i]:+.3f}' for i in idx])}

Student Input Values:
{chr(10).join([f'- {feat}: {val}' for feat, val in user_input.items()])}

Note: This assessment is generated by an AI model and should be used as a decision support tool.
"""
                
                st.download_button(
                    label="📄 Download Full Report",
                    data=results_text,
                    file_name=f"dropout_risk_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download a detailed report of this assessment"
                )
                
                st.success("✅ Assessment complete! Download the report or share with counselors.")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Try different input values or check the model.")
    
    # ========== FAIRNESS DASHBOARD ==========
    with st.expander("⚖️ Fairness Dashboard", expanded=False):
        st.markdown("### Model Performance Across Groups")
        
        tab1, tab2 = st.tabs(["Gender Analysis", "Quick Stats"])
        
        with tab1:
            if 'Gender' in df.columns:
                # Simplified fairness analysis
                male_data = df[df['Gender'] == 0]
                female_data = df[df['Gender'] == 1]
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Male Students**")
                    if len(male_data) > 0:
                        st.write(f"Count: {len(male_data)}")
                        if 'target' in male_data.columns:
                            dropout_rate = (male_data['target'] == 'Dropout').mean()
                            st.write(f"Dropout rate: {dropout_rate:.1%}")
                
                with col_g2:
                    st.markdown("**Female Students**")
                    if len(female_data) > 0:
                        st.write(f"Count: {len(female_data)}")
                        if 'target' in female_data.columns:
                            dropout_rate = (female_data['target'] == 'Dropout').mean()
                            st.write(f"Dropout rate: {dropout_rate:.1%}")
            else:
                st.info("Gender data not available in dataset")
        
        with tab2:
            st.markdown("**Model Performance**")
            st.write(f"Test Accuracy: {accuracy:.2%}")
            st.write(f"Features Used: {len(feature_cols)}")
            st.write(f"Training Samples: ~{len(df) * 0.8:.0f}")
            st.write(f"Validation Samples: ~{len(df) * 0.2:.0f}")
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>🎓 <strong>Dropout Risk Advisor</strong> | AI-powered student success tool</p>
        <p>⚠️ This tool provides predictions for decision support. Always combine with human judgment.</p>
    </div>
    """, unsafe_allow_html=True)

# ========== RUN APP ==========
if __name__ == "__main__":
    main()
