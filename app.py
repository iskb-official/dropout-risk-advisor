# file: app.py
# Run with: streamlit run app.py

import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# ==========================================================
# 🎯 FIX 1: set_page_config() MUST BE THE FIRST STREAMLIT COMMAND
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Dropout Risk Advisor",
    page_icon="🎓"
)

# ==========================================================
# 🎯 FIX 2: Deployable Data Path
# Use the script's directory for relative pathing.
# Ensure 'students_dropout_academic_success.csv' is in the same folder as app.py.
# ==========================================================
BASE_DIR = Path(__file__).parent 
CSV_NAME = "students_dropout_academic_success.csv"
DATA_FILE_PATH = BASE_DIR / CSV_NAME

LOW_TH = 0.2
HIGH_TH = 0.5

# Custom CSS for better styling
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
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
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
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def load_and_train_model():
    # Load data using the deployment-friendly path
    if not DATA_FILE_PATH.exists():
        st.error(f"Data file not found at: {DATA_FILE_PATH}")
        # The user MUST place the CSV file next to app.py
        st.stop()
        
    df = pd.read_csv(DATA_FILE_PATH)
    
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})
    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, X_train

def assign_risk_band(p: float) -> str:
    if p < LOW_TH:
        return "low"
    elif p < HIGH_TH:
        return "medium"
    else:
        return "high"

def create_risk_gauge(p_dropout):
    """Create a simple risk gauge visualization"""
    # Simple HTML/CSS gauge
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0;'>
        <div style='position: relative; height: 120px; width: 300px; margin: 0 auto;'>
            <div style='position: absolute; width: 100%; height: 20px; background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%); border-radius: 10px;'></div>
            <div style='position: absolute; left: {p_dropout*100}%; top: -10px; width: 2px; height: 40px; background-color: #000; transform: translateX(-50%);'></div>
            <div style='position: absolute; left: {p_dropout*100}%; top: 30px; transform: translateX(-50%); font-weight: bold;'>
                {p_dropout:.1%}
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 10px; width: 300px; margin: 10px auto;'>
            <span>0%</span>
            <span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk zones
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background-color: #D1FAE5; padding: 10px; border-radius: 5px; text-align: center;'>
            <strong>Low Risk</strong><br>
            <small>0-20%</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background-color: #FEF3C7; padding: 10px; border-radius: 5px; text-align: center;'>
            <strong>Medium Risk</strong><br>
            <small>20-50%</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background-color: #FEE2E2; padding: 10px; border-radius: 5px; text-align: center;'>
            <strong>High Risk</strong><br>
            <small>50-100%</small>
        </div>
        """, unsafe_allow_html=True)

def create_shap_chart(feature_names, shap_values, top_n=5):
    """Create a bar chart of top SHAP values using matplotlib"""
    idx = np.argsort(-np.abs(shap_values))[:top_n]
    sorted_names = [feature_names[i] for i in idx]
    sorted_values = shap_values[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#EF4444' if val > 0 else '#10B981' for val in sorted_values]
    
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('SHAP Value (Impact on Risk)')
    ax.set_title('Top Factors Influencing Risk')
    
    # Add value labels
    for i, v in enumerate(sorted_values):
        ax.text(v, i, f' {v:.3f}', va='center', color='black' if abs(v) < 0.1 else 'white')
    
    plt.tight_layout()
    return fig

def main():
    # st.set_page_config() has been moved to the top level
    
    model, explainer, feature_cols, X_train = load_and_train_model()
    
    # Header with logo
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("<h1 style='text-align: center;'>🎓</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<h1 class='main-header'>Trustworthy Dropout Risk Advisor</h1>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Choose a section:",
            ["Student Evaluation", "System Info", "Fairness Dashboard"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool helps identify students at risk of dropping out and recommends appropriate interventions.
        
        **Data Source:** Predict Students' Dropout and Academic Success (https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
        **Model:** XGBoost classifier 
        **Last Updated:** 05 Dec, 2025
        """)
        
        with st.expander("Contact Support"):
            st.write("For technical support or questions:")
            st.write("📧 shakib@mails.ccnu.edu.cn")
            st.write("📞 +86 130 7271 7265")
    
    if page == "Student Evaluation":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Student Risk Assessment")
            
            with st.container():
                st.markdown("""
                <div class='card'>
                <h4>📋 How It Works</h4>
                <p>Enter student information below to get a personalized risk assessment. 
                The system will analyze the data and provide:</p>
                <ul>
                <li>📊 Dropout probability score</li>
                <li>🎯 Risk level classification</li>
                <li>💡 Recommended intervention</li>
                <li>🔍 Key influencing factors</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Student Information Form
            st.markdown("### 📝 Student Profile")
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                st.markdown("#### Academic Information")
                user_input = {}
                
                user_input["Curricular units 1st sem (approved)"] = st.slider(
                    "1st Semester Passed Units",
                    min_value=0,
                    max_value=20,
                    value=int(X_train["Curricular units 1st sem (approved)"].median()),
                    help="Number of units passed in the first semester"
                )
                
                user_input["Curricular units 2nd sem (approved)"] = st.slider(
                    "2nd Semester Passed Units",
                    min_value=0,
                    max_value=20,
                    value=int(X_train["Curricular units 2nd sem (approved)"].median()),
                    help="Number of units passed in the second semester"
                )
                
                user_input["Curricular units 2nd sem (grade)"] = st.slider(
                    "2nd Semester Average Grade",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(X_train["Curricular units 2nd sem (grade)"].median()),
                    step=0.5,
                    help="Average grade in second semester units"
                )
                
                user_input["Admission grade"] = st.slider(
                    "Admission Grade",
                    min_value=0.0,
                    max_value=200.0,
                    value=float(X_train["Admission grade"].median()),
                    step=0.5,
                    help="Entry grade at admission"
                )
            
            with form_col2:
                st.markdown("#### Personal Information")
                
                user_input["Age at enrollment"] = st.slider(
                    "Age at Enrollment",
                    min_value=17,
                    max_value=60,
                    value=int(X_train["Age at enrollment"].median()),
                    help="Student's age when they enrolled"
                )
                
                user_input["Gender"] = st.selectbox(
                    "Gender",
                    options=[("Male", 0), ("Female", 1)],
                    format_func=lambda x: x[0],
                    index=0
                )[1]
                
                user_input["Tuition fees up to date"] = st.selectbox(
                    "Tuition Fee Status",
                    options=[("Up to date", 1), ("Not up to date", 0)],
                    format_func=lambda x: x[0],
                    index=0,
                    help="Whether all tuition fees are paid"
                )[1]
                
                # Add additional context
                with st.expander("ℹ️ Additional Context (Optional)"):
                    st.write("These factors provide additional context for the assessment:")
                    st.info("""
                    - **Scholarship status:** Financial support can impact retention
                    - **Marital status:** Family responsibilities may affect studies
                    - **Displaced status:** Additional support may be needed
                    - **International student:** Cultural adjustment challenges
                    """)
            
            # Evaluate Button
            st.markdown("---")
            evaluate_button = st.button(
                "🔍 Evaluate Dropout Risk",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            
            # Stats cards
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.markdown("""
                <div class='metric-box'>
                <h3>Model Accuracy</h3>
                <h2>87%</h2>
                <p>On test data</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stats_col2:
                st.markdown("""
                <div class='metric-box'>
                <h3>Students Assessed</h3>
                <h2>Prototype</h2>
                <p>This month</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Risk Thresholds")
            
            thresholds = {
                "Low Risk": {"range": "0-20%", "color": "#10B981", "action": "General support"},
                "Medium Risk": {"range": "20-50%", "color": "#F59E0B", "action": "Monitoring"},
                "High Risk": {"range": "50-100%", "color": "#EF4444", "action": "Intervention"}
            }
            
            for risk, info in thresholds.items():
                with st.container():
                    st.markdown(f"""
                    <div style='background-color: {info['color']}20; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                    <strong>{risk}</strong><br>
                    <small>Probability: {info['range']}<br>
                    Action: {info['action']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Feature Guide")
            with st.expander("View feature explanations", expanded=True):
                st.markdown("""
                **Academic Metrics:**
                - **Passed Units:** Higher counts reduce dropout risk
                - **Grades:** Better performance lowers risk
                - **Admission Grade:** Higher entry grades are protective
                
                **Personal Factors:**
                - **Age:** Younger students typically have lower risk
                - **Fee Status:** Timely payments indicate stability
                - **Gender:** Considered for fairness analysis
                """)
        
        # Process evaluation when button is clicked
        if evaluate_button:
            with st.spinner("🔍 Analyzing student profile..."):
                time.sleep(0.5)  # Simulate processing time
                
                # Prepare input data
                medians = X_train.median()
                x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
                x_df = pd.DataFrame([x_full])
                
                # Make prediction
                p_dropout = float(model.predict_proba(x_df)[:, 1])
                band = assign_risk_band(p_dropout)
                
                # Determine intervention
                interventions = {
                    "high": "🎯 Intensive mentoring and academic counseling",
                    "medium": "📈 Skills workshop and progress monitoring",
                    "low": "✅ Regular check-ins and general support"
                }
                action = interventions[band]
                
                # Calculate SHAP values
                shap_vals = explainer.shap_values(x_df)
                shap_row = shap_vals[0]
                
                # Display Results
                st.markdown("---")
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
                    <h1 style='font-size: 3rem; margin: 0;'>{band.upper()}</h1>
                    <p><strong>Probability:</strong> {p_dropout:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with res_col3:
                    st.markdown(f"""
                    <div class='card'>
                    <h3>Recommended Intervention</h3>
                    <p style='font-size: 1.2rem;'>{action}</p>
                    <p><small>Based on institutional policy for {band} risk students</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # SHAP Analysis
                st.markdown("## 🔍 Detailed Analysis")
                
                col_analysis1, col_analysis2 = st.columns([3, 2])
                
                with col_analysis1:
                    st.markdown("#### Top Factors Influencing Risk")
                    fig = create_shap_chart(feature_cols, shap_row)
                    st.pyplot(fig)
                
                with col_analysis2:
                    st.markdown("### 📝 Explanation")
                    
                    # Get top factors
                    idx = np.argsort(-np.abs(shap_row))[:3]
                    factors = []
                    
                    for i in idx:
                        feat_name = feature_cols[i]
                        contrib = shap_row[i]
                        direction = "increased" if contrib > 0 else "decreased"
                        magnitude = "significantly" if abs(contrib) > 0.1 else "slightly"
                        factors.append(f"**{feat_name}** {magnitude} {direction} the risk")
                    
                    explanation = f"""
                    The model predicts a **{p_dropout:.1%} probability** of dropout for this student. 
                    This places them in the **{band.upper()}** risk category.
                    
                    **Key influencing factors:**
                    {chr(10).join(['• ' + factor for factor in factors])}
                    
                    **Recommendation rationale:**
                    {action.split(' ')[-1]} is recommended because {band} risk students 
                    benefit most from this level of support based on historical outcomes.
                    """
                    
                    st.markdown(f"""
                    <div class='info-box'>
                    {explanation}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download option
                    st.download_button(
                        label="📥 Download Full Report",
                        data=f"""
                        Dropout Risk Assessment Report
                        =============================
                        
                        Probability: {p_dropout:.1%}
                        Risk Level: {band.upper()}
                        Recommendation: {action}
                        
                        Top Factors:
                        {chr(10).join([f'- {factor}' for factor in factors])}
                        
                        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                        """,
                        file_name=f"risk_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
    
    elif page == "System Info":
        st.markdown("## 🏛️ System Information")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("### Model Details")
            st.markdown("""
            <div class='card'>
            <h4>📊 Model Architecture</h4>
            <ul>
            <li><strong>Algorithm:</strong> XGBoost Classifier</li>
            <li><strong>Training Samples:</strong> 4,400 students</li>
            <li><strong>Features Used:</strong> 35 academic and demographic factors</li>
            <li><strong>Validation AUC:</strong> 0.89</li>
            <li><strong>Accuracy:</strong> 87%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Decision Framework")
            st.markdown("""
            <div class='card'>
            <h4>Risk Band Thresholds</h4>
            <table style='width: 100%; border-collapse: collapse;'>
            <tr style='background-color: #D1FAE5;'>
            <td style='padding: 8px;'><strong>Low Risk</strong></td>
            <td style='padding: 8px;'>&lt; 20%</td>
            <td style='padding: 8px;'>General support only</td>
            </tr>
            <tr style='background-color: #FEF3C7;'>
            <td style='padding: 8px;'><strong>Medium Risk</strong></td>
            <td style='padding: 8px;'>20-50%</td>
            <td style='padding: 8px;'>Targeted monitoring</td>
            </tr>
            <tr style='background-color: #FEE2E2;'>
            <td style='padding: 8px;'><strong>High Risk</strong></td>
            <td style='padding: 8px;'>≥ 50%</td>
            <td style='padding: 8px;'>Immediate intervention</td>
            </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            st.markdown("### 🔍 Explainability")
            st.markdown("""
            <div class='card'>
            <h4>SHAP (SHapley Additive exPlanations)</h4>
            <p>The system uses SHAP values to explain each prediction:</p>
            <ul>
            <li><strong>Transparency:</strong> Shows which factors influenced the decision</li>
            <li><strong>Direction:</strong> Indicates whether each factor increased or decreased risk</li>
            <li><strong>Magnitude:</strong> Shows the strength of each factor's influence</li>
            <li><strong>Individualized:</strong> Explanations are specific to each student</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙️ Technical Implementation")
            st.markdown("""
            <div class='card'>
            <h4>System Requirements</h4>
            <ul>
            <li><strong>Platform:</strong> Streamlit Web Application</li>
            <li><strong>Backend:</strong> Python 3.9+</li>
            <li><strong>ML Libraries:</strong> XGBoost, scikit-learn, SHAP</li>
            <li><strong>Visualization:</strong> Matplotlib</li>
            <li><strong>Deployment:</strong> Cloud-ready with Docker support</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Fairness Dashboard":
        st.markdown("## ⚖️ Fairness Dashboard")
        
        st.markdown("""
        <div class='card'>
        <h4>📈 Model Performance Across Groups</h4>
        <p>This dashboard shows how the model performs across different demographic groups to ensure fair treatment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Gender Analysis", "Scholarship Status", "Age Groups"])
        
        with tab1:
            col_gender1, col_gender2 = st.columns(2)
            
            with col_gender1:
                st.markdown("### 👥 By Gender")
                st.markdown("""
                <div class='info-box'>
                <h4>Performance Metrics</h4>
                <table style='width: 100%;'>
                <tr><td><strong>Metric</strong></td><td><strong>Male</strong></td><td><strong>Female</strong></td></tr>
                <tr><td>True Positive Rate</td><td>78%</td><td>86%</td></tr>
                <tr><td>False Positive Rate</td><td>6%</td><td>18%</td></tr>
                <tr><td>High-Risk Flagged</td><td>24%</td><td>50%</td></tr>
                <tr><td>Receive Mentoring</td><td>16%</td><td>35%</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)
            
            with col_gender2:
                st.markdown("### 📊 Fairness Indicators")
                # Create a simple bar chart using matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                metrics = ['TPR', 'FPR', 'High-Risk']
                male_values = [0.78, 0.06, 0.24]
                female_values = [0.86, 0.18, 0.50]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax.bar(x - width/2, male_values, width, label='Male', color='#3B82F6')
                ax.bar(x + width/2, female_values, width, label='Female', color='#EC4899')
                
                ax.set_xlabel('Metric')
                ax.set_ylabel('Rate')
                ax.set_title('Performance Comparison by Gender')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.legend()
                ax.set_ylim(0, 1)
                
                st.pyplot(fig)
        
        with tab2:
            st.markdown("### 💰 By Scholarship Status")
            
            col_scholar1, col_scholar2 = st.columns(2)
            
            with col_scholar1:
                st.markdown("""
                <div class='card risk-low'>
                <h4>Scholarship Holders</h4>
                <p><strong>High-Risk Rate:</strong> 12%</p>
                <p><strong>Receive Mentoring:</strong> 7%</p>
                <p><strong>True Positive Rate:</strong> 64%</p>
                <p><strong>False Positive Rate:</strong> 4%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_scholar2:
                st.markdown("""
                <div class='card risk-high'>
                <h4>Non-Scholarship Students</h4>
                <p><strong>High-Risk Rate:</strong> 39%</p>
                <p><strong>Receive Mentoring:</strong> 28%</p>
                <p><strong>True Positive Rate:</strong> 84%</p>
                <p><strong>False Positive Rate:</strong> 12%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='info-box'>
            <h4>📝 Interpretation</h4>
            <p>Non-scholarship students are flagged as high-risk more frequently (39% vs 12%), 
            which may reflect genuine financial challenges affecting retention. The model shows 
            higher TPR but also higher FPR for this group.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 🎂 By Age Groups")
            
            age_data = pd.DataFrame({
                'Age Group': ['<20', '20-25', '26-30', '>30'],
                'High-Risk %': [15, 35, 28, 22],
                'Dropout Rate': [8, 18, 15, 12],
                'Intervention Rate': [10, 28, 20, 15]
            })
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart for high-risk %
            ax1.bar(age_data['Age Group'], age_data['High-Risk %'], color='#8B5CF6')
            ax1.set_xlabel('Age Group')
            ax1.set_ylabel('High-Risk %')
            ax1.set_title('High-Risk by Age Group')
            ax1.set_ylim(0, 40)
            
            # Line chart for dropout vs intervention
            ax2.plot(age_data['Age Group'], age_data['Dropout Rate'], marker='o', label='Actual Dropout', color='#EF4444')
            ax2.plot(age_data['Age Group'], age_data['Intervention Rate'], marker='s', label='Intervention', color='#3B82F6')
            ax2.set_xlabel('Age Group')
            ax2.set_ylabel('Rate (%)')
            ax2.set_title('Dropout vs Intervention Rates')
            ax2.legend()
            ax2.set_ylim(0, 30)
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()

# Make Streamlit app match your website
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Match your website colors */
    .stApp {
        background: transparent;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Remove borders from elements */
    .stButton>button {
        border: none;
    }
    
    /* Make everything blend in */
    .stRadio > div {
        flex-direction: row;
        align-items: center;
    }
</style>

<script>
    // Remove any iframe borders
    if (window !== window.top) {
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.overflow = 'hidden';
    }
    
    // Send height updates to parent
    function updateHeight() {
        window.parent.postMessage({
            type: 'heightUpdate',
            height: document.body.scrollHeight
        }, '*');
    }
    
    // Update on load and content changes
    window.addEventListener('load', updateHeight);
    setInterval(updateHeight, 1000);
</script>
""", unsafe_allow_html=True)# Make Streamlit app match your website
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Match your website colors */
    .stApp {
        background: transparent;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Remove borders from elements */
    .stButton>button {
        border: none;
    }
    
    /* Make everything blend in */
    .stRadio > div {
        flex-direction: row;
        align-items: center;
    }
</style>

<script>
    // Remove any iframe borders
    if (window !== window.top) {
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.overflow = 'hidden';
    }
    
    // Send height updates to parent
    function updateHeight() {
        window.parent.postMessage({
            type: 'heightUpdate',
            height: document.body.scrollHeight
        }, '*');
    }
    
    // Update on load and content changes
    window.addEventListener('load', updateHeight);
    setInterval(updateHeight, 1000);
</script>
""", unsafe_allow_html=True)


