Here is the cleaned and corrected script. I've fixed the broken markdown, indentation errors, malformed f-strings, and the split regex that was causing syntax issues. The logic and structure remain identical to your original intent.

```python
# file: app.py
# Run with: streamlit run app.py

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# ==========================================================
# 🎯 PAGE CONFIG MUST BE FIRST
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Dropout Risk DSS",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

# ==========================================================
# 🎯 PATHING & CONSTANTS
# ==========================================================
BASE_DIR = Path(__file__).parent
CSV_NAME = "students_dropout_academic_success.csv"
DATA_FILE_PATH = BASE_DIR / CSV_NAME

LOW_TH = 0.20
HIGH_TH = 0.50
CAPACITY_LIMIT = 200

# ==========================================================
# 🎨 MODERN CSS
# ==========================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #457B9D;
        margin-bottom: 1.5rem;
    }
    .tier-low  { background-color: #D8F3DC; color: #1B4332; padding: 1rem; border-radius: 8px; }
    .tier-med  { background-color: #FFF3CD; color: #856404; padding: 1rem; border-radius: 8px; }
    .tier-high { background-color: #F8D7DA; color: #842029; padding: 1rem; border-radius: 8px; }
    .metric-card {
        background-color: #F1FAEE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1D3557;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 🧠 MODEL & DATA CACHING
# ==========================================================
@st.cache_resource(show_spinner=True)
def load_and_train_model():
    if not DATA_FILE_PATH.exists():
        st.error(f"Data file not found at: {DATA_FILE_PATH}")
        st.stop()

    df = pd.read_csv(DATA_FILE_PATH)

    # EXACT PAPER METHODOLOGY: Drop "Enrolled" to remove right-censored label noise
    df = df[df["target"] != "Enrolled"].copy()
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0})

    # Clean columns for XGBoost (strip characters that break feature names)
    regex = re.compile(r"[\[\]<>]", re.IGNORECASE)
    df.columns = [regex.sub("_", str(col)) for col in df.columns]

    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    # 80/20 Split matching paper's test cohort size (~726 students)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train Engine
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, X_train


def assign_risk_band(p: float) -> str:
    if p < LOW_TH:
        return "Low Risk"
    elif p < HIGH_TH:
        return "Medium Risk"
    else:
        return "High Risk"


def get_intervention(band: str) -> str:
    if band == "High Risk":
        return "Intensive Mentoring + Counseling"
    elif band == "Medium Risk":
        return "Skills Workshops + Progress Monitoring"
    else:
        return "General Academic Support"


# ==========================================================
# 📊 INTERACTIVE PLOTLY COMPONENTS
# ==========================================================
def plot_gauge(probability, band):
    color = (
        "#2A9D8F"
        if band == "Low Risk"
        else "#E9C46A"
        if band == "Medium Risk"
        else "#E63946"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {"size": 40, "color": color, "weight": "bold"},
            },
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 20], "color": "rgba(42, 157, 143, 0.15)"},
                    {"range": [20, 50], "color": "rgba(233, 196, 106, 0.15)"},
                    {"range": [50, 100], "color": "rgba(230, 57, 70, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_shap_waterfall(feature_names, shap_values, base_value=0.124):
    """Plotly horizontal bar chart mimicking SHAP directional attributions"""
    idx = np.argsort(np.abs(shap_values))[-5:]  # Top 5
    names = [feature_names[i] for i in idx]
    vals = shap_values[idx]

    colors = ["#E63946" if v > 0 else "#2A9D8F" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in vals],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Top Factors Driving Risk (SHAP Attributions)",
        xaxis_title="Impact on Model Output",
        yaxis_title="",
        height=300,
        margin=dict(l=10, r=30, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor="#1D3557",
            zerolinewidth=2,
        ),
    )
    return fig


# ==========================================================
# 🚀 MAIN APP
# ==========================================================
def main():
    model, explainer, feature_cols, X_train = load_and_train_model()

    st.markdown(
        '<div class="main-header">🎓 Capacity-Aware Decision Support System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Interpretable Student Retention Prototype</div>',
        unsafe_allow_html=True,
    )

    tab_eval, tab_info, tab_fairness = st.tabs(
        [
            "🎯 Student Risk Assessment",
            "🏛️ System Architecture",
            "⚖️ Fairness Mitigation Audit",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: EVALUATION
    # ---------------------------------------------------------
    with tab_eval:
        col_input, col_results = st.columns([1.2, 2])

        with col_input:
            st.markdown("---")
            st.markdown("### 📝 Student Profile")

            user_input = {}

            st.markdown("##### Academic Indicators")
            user_input["Curricular units 1st sem (approved)"] = st.slider(
                "1st Semester Passed Units",
                0,
                20,
                int(X_train["Curricular units 1st sem (approved)"].median()),
            )
            user_input["Curricular units 2nd sem (approved)"] = st.slider(
                "2nd Semester Passed Units",
                0,
                20,
                int(X_train["Curricular units 2nd sem (approved)"].median()),
            )
            user_input["Curricular units 2nd sem (grade)"] = st.slider(
                "2nd Semester Average Grade",
                0.0,
                20.0,
                float(X_train["Curricular units 2nd sem (grade)"].median()),
                0.5,
            )

            st.markdown("##### Socio-Financial Indicators")
            user_input["Tuition fees up to date"] = st.selectbox(
                "Tuition Fee Status",
                options=[("Up to date", 1), ("Overdue", 0)],
                format_func=lambda x: x[0],
            )[1]
            user_input["Scholarship holder"] = st.selectbox(
                "Scholarship Holder",
                options=[("Yes", 1), ("No", 0)],
                format_func=lambda x: x[0],
            )[1]
            user_input["Gender"] = st.selectbox(
                "Gender (UCI Encoding)",
                options=[("Male", 1), ("Female", 0)],
                format_func=lambda x: x[0],
            )[1]
            user_input["Age at enrollment"] = st.number_input(
                "Age at Enrollment", 17, 65, 20
            )

            evaluate = st.button(
                "🔍 Generate Diagnostic Prediction", use_container_width=True
            )
            st.markdown("---")

        with col_results:
            if evaluate:
                with st.spinner(
                    "Executing XGBoost and computing TreeSHAP matrices..."
                ):
                    time.sleep(0.6)  # UX pause

                    # Impute missing with medians to form full 36-feature vector
                    medians = X_train.median()
                    x_full = {
                        feat: user_input.get(feat, float(medians[feat]))
                        for feat in feature_cols
                    }
                    x_df = pd.DataFrame([x_full])

                    # Inference
                    p_dropout = float(model.predict_proba(x_df)[:, 1])
                    band = assign_risk_band(p_dropout)
                    action = get_intervention(band)

                    shap_vals = explainer.shap_values(x_df)[0]

                    # Render Results
                    tier_class = (
                        "tier-low"
                        if band == "Low Risk"
                        else "tier-med"
                        if band == "Medium Risk"
                        else "tier-high"
                    )

                    st.markdown(
                        f"""
                        <div class="{tier_class}">
                            <h3 style="margin-top:0;">Diagnostic Output</h3>
                            <p><strong>Risk Stratification:</strong> {band.upper()}</p>
                            <p style="margin-bottom:0;"><strong>Prescriptive Action:</strong> {action}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_gauge, col_shap = st.columns([1, 1.5])
                    with col_gauge:
                        st.markdown("---")
                        st.markdown("**Attrition Probability**")
                        st.plotly_chart(
                            plot_gauge(p_dropout, band),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                        st.markdown("---")

                    with col_shap:
                        st.markdown("---")
                        st.plotly_chart(
                            plot_shap_waterfall(feature_cols, shap_vals),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                        st.markdown("---")
            else:
                st.info(
                    "👈 Enter the student's metrics and click "
                    "'Generate Diagnostic Prediction' to view "
                    "capacity-aware interventions and SHAP attributions."
                )

    # ---------------------------------------------------------
    # TAB 2: SYSTEM ARCHITECTURE
    # ---------------------------------------------------------
    with tab_info:
        col_m1, col_m2, col_m3 = st.columns(3)

        col_m1.markdown(
            """
            <div class="metric-card">
                <h4>Validation Engine</h4>
                <p><strong>XGBoost</strong></p>
                <p>5-Fold Stratified CV</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_m2.markdown(
            """
            <div class="metric-card">
                <h4>Cross-Modality AUC</h4>
                <p><strong>&gt; 0.950</strong></p>
                <p>Traditional, LMS, xAPI, MOOC</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_m3.markdown(
            """
            <div class="metric-card">
                <h4>Calibration Metric</h4>
                <p><strong>0.058</strong></p>
                <p>Brier Score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### Mathematical Capacity Constraint (C=200)

            Unlike standard predictive models that output a vacuum probability,
            this Decision Support System utilizes a mathematically defined
            objective function to allocate resources:

            ```
            max ∑ (p_i · I(p_i ≥ T_H)) · a_i    s.t.    ∑ a_i ≤ C
            ```

            This explicitly forces the algorithm to prioritize the top 200
            high-risk cases for Intensive Mentoring, dynamically routing
            capacity overflow to scalable workshops.
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # TAB 3: FAIRNESS MITIGATION
    # ---------------------------------------------------------
    with tab_fairness:
        st.markdown(
            """
            ### Active Equal Opportunity Mitigation

            Rather than passively auditing bias, the framework actively
            enforces Equal Opportunity by optimizing group-specific
            probability thresholds (T_H,g). This neutralizes historical
            demographic disparities in intervention allocation.
            """,
            unsafe_allow_html=True,
        )

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown(
                """
                #### Unmitigated Baseline (Global Threshold)

                A global threshold results in an unacceptable sensitivity
                gap between financial groups, disproportionately failing
                to flag at-risk scholarship students.

                * **Non-Scholarship TPR:** 0.907
                * **Scholarship TPR:** 0.885
                * **Sensitivity Gap (ΔTPR): 0.022**
                """,
                unsafe_allow_html=True,
            )

        with col_f2:
            st.markdown(
                """
                #### Mitigated State (Group Thresholds)

                Applying mathematically optimized thresholds
                (Scholarship: 0.767, Non-Scholarship: 0.778)
                equalizes the True Positive Rates.

                * **Non-Scholarship TPR:** 0.849
                * **Scholarship TPR:** 0.846
                * **Sensitivity Gap (ΔTPR): 0.003**
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
```
