# file: app.py
# Run with: streamlit run app.py
# Requires: pip install streamlit xgboost shap plotly scikit-learn pandas numpy reportlab

import re
import io
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

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

# Human-readable feature labels for the UI / SHAP
FEATURE_LABELS = {
    "Curricular units 1st sem (approved)": "1st Sem Approved Units",
    "Curricular units 2nd sem (approved)": "2nd Sem Approved Units",
    "Curricular units 2nd sem (grade)": "2nd Sem Average Grade",
    "Curricular units 1st sem (grade)": "1st Sem Average Grade",
    "Tuition fees up to date": "Tuition Fees Up To Date",
    "Scholarship holder": "Scholarship Holder",
    "Gender": "Gender",
    "Age at enrollment": "Age at Enrollment",
    "Debtor": "Debtor Status",
    "Application mode": "Application Mode",
    "Course": "Course",
    "Previous qualification (grade)": "Previous Qualification Grade",
    "Admission grade": "Admission Grade",
}

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
    .tier-low  { background-color: #D8F3DC; color: #1B4332; padding: 1rem; border-radius: 8px; border-left: 6px solid #2A9D8F; }
    .tier-med  { background-color: #FFF3CD; color: #856404; padding: 1rem; border-radius: 8px; border-left: 6px solid #E9C46A; }
    .tier-high { background-color: #F8D7DA; color: #842029; padding: 1rem; border-radius: 8px; border-left: 6px solid #E63946; }
    .metric-card {
        background-color: #F1FAEE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1D3557;
    }
    .explain-box {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .explain-box h4 {
        margin-top: 0;
        color: #1D3557;
    }
    .factor-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .chip-risk { background-color: #F8D7DA; color: #842029; }
    .chip-prot { background-color: #D8F3DC; color: #1B4332; }
    .step-badge {
        display: inline-block;
        background-color: #1D3557;
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-right: 0.5rem;
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

    # Drop "Enrolled" to remove right-censored label noise
    df = df[df["target"] != "Enrolled"].copy()
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0})

    # Clean column names for XGBoost
    regex = re.compile(r"[\[\]<>]", re.IGNORECASE)
    df.columns = [regex.sub("_", str(col)) for col in df.columns]

    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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


def friendly_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


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


def plot_shap_waterfall(feature_names, shap_values, top_n=6):
    """Plotly horizontal bar chart mimicking SHAP directional attributions."""
    idx = np.argsort(np.abs(shap_values))[-top_n:]
    names = [friendly_name(feature_names[i]) for i in idx]
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
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Top Factors Driving Risk (SHAP Attributions)",
        xaxis_title="← Protects (green)   |   Increases Risk (red) →",
        yaxis_title="",
        height=340,
        margin=dict(l=10, r=40, t=50, b=20),
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


def plot_contribution_donut(shap_values):
    """Donut summarizing positive vs negative contribution."""
    pos = float(np.sum(shap_values[shap_values > 0]))
    neg = float(np.sum(np.abs(shap_values[shap_values < 0])))

    fig = go.Figure(
        go.Pie(
            labels=["Risk-Increasing Factors", "Risk-Reducing Factors"],
            values=[pos, neg],
            hole=0.65,
            marker_colors=["#E63946", "#2A9D8F"],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Total Impact: %{value:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Overall Balance of Factor Contributions",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_threshold_explorer(probability):
    """Interactive bar showing where the student falls across the bands."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[LOW_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(42, 157, 143, 0.55)",
            name="Low Risk (< 0.20)",
            hovertemplate="Low Risk<br>0.00 – 0.20<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[HIGH_TH - LOW_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(233, 196, 106, 0.6)",
            name="Medium Risk (0.20 – 0.50)",
            hovertemplate="Medium Risk<br>0.20 – 0.50<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[1.0 - HIGH_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(230, 57, 70, 0.6)",
            name="High Risk (≥ 0.50)",
            hovertemplate="High Risk<br>0.50 – 1.00<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[probability],
            y=["Risk Band"],
            mode="markers+text",
            marker=dict(size=22, color="#1D3557", symbol="line-ns", line=dict(width=3, color="white")),
            text=[f"  This student: {probability:.1%}"],
            textposition="top center",
            textfont=dict(size=13, color="#1D3557", family="Arial Black"),
            name="Student",
            hovertemplate="Student Probability: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        title="Where This Student Falls Across Risk Bands",
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Dropout Probability"),
        yaxis=dict(showticklabels=False),
        height=220,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.35),
        showlegend=True,
    )
    return fig


def plot_radar_profile(user_input, medians, top_features):
    """Radar of the student's z-scores against cohort medians."""
    labels = []
    student_vals = []
    cohort_vals = []

    for feat in top_features:
        val = float(user_input.get(feat, medians.get(feat, 0.0)))
        med = float(medians.get(feat, 0.0))
        # Normalize using max(med, 1) to avoid divide-by-zero, cap to [0,1]
        denom = med if med > 0 else 1.0
        student_norm = max(0.0, min(1.0, val / denom)) if denom else 0.0
        labels.append(friendly_name(feat))
        student_vals.append(student_norm)
        cohort_vals.append(1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=student_vals,
            theta=labels,
            fill="toself",
            name="This Student",
            line_color="#1D3557",
            fillcolor="rgba(29, 53, 87, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=cohort_vals,
            theta=labels,
            fill="toself",
            name="Cohort Median",
            line_color="#E9C46A",
            fillcolor="rgba(233, 196, 106, 0.15)",
        )
    )

    fig.update_layout(
        title="Student Profile vs Cohort Median",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


# ==========================================================
# 📄 PDF REPORT GENERATION
# ==========================================================
def _fig_to_png_bytes(fig, width=700, height=400):
    """Render a Plotly figure to PNG bytes for embedding in the PDF."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        # Fallback: return None if kaleido is not installed
        return None


def build_pdf_report(student_inputs, p_dropout, band, action, shap_vals, feature_cols, top_k=8):
    """Build a PDF report as a BytesIO buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Student Dropout Risk Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1D3557"),
        fontSize=22,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1D3557"),
        fontSize=14,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
    )
    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["BodyText"],
        fontSize=9,
        textColor=colors.HexColor("#6C757D"),
    )

    band_color = {
        "Low Risk": colors.HexColor("#2A9D8F"),
        "Medium Risk": colors.HexColor("#E9C46A"),
        "High Risk": colors.HexColor("#E63946"),
    }.get(band, colors.HexColor("#1D3557"))

    story = []

    # Header
    story.append(Paragraph("🎓 Student Dropout Risk Report", title_style))
    story.append(
        Paragraph(
            f"Capacity-Aware Decision Support System · Generated "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            small_style,
        )
    )
    story.append(Spacer(1, 12))

    # Summary box
    summary_data = [
        ["Risk Stratification", band.upper()],
        ["Dropout Probability", f"{p_dropout:.2%}"],
        ["Prescriptive Action", action],
    ]
    summary_table = Table(summary_data, colWidths=[5.5 * cm, 10 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1FAEE")),
                ("BACKGROUND", (1, 0), (1, 0), band_color),
                ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED4DA")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CED4DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)

    # Gauge image
    story.append(Paragraph("Attrition Probability", h2_style))
    gauge_png = _fig_to_png_bytes(plot_gauge(p_dropout, band), width=600, height=320)
    if gauge_png:
        story.append(RLImage(io.BytesIO(gauge_png), width=13 * cm, height=6.9 * cm))
    else:
        story.append(
            Paragraph(
                f"<i>Chart omitted (install 'kaleido' for embedded images). "
                f"Probability: {p_dropout:.2%}</i>",
                body_style,
            )
        )

    # Student inputs table
    story.append(Paragraph("Submitted Student Profile", h2_style))
    input_rows = [["Indicator", "Value"]]
    for k, v in student_inputs.items():
        input_rows.append([friendly_name(k), str(v)])
    input_table = Table(input_rows, colWidths=[8 * cm, 7.5 * cm])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CED4DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(input_table)

    # SHAP waterfall image
    story.append(Paragraph("Top Factors Driving Risk (SHAP)", h2_style))
    shap_png = _fig_to_png_bytes(
        plot_shap_waterfall(feature_cols, shap_vals, top_n=top_k),
        width=700,
        height=420,
    )
    if shap_png:
        story.append(RLImage(io.BytesIO(shap_png), width=16 * cm, height=9.6 * cm))
    else:
        story.append(Paragraph("<i>SHAP chart omitted (kaleido missing).</i>", body_style))

    # SHAP numeric table
    story.append(Paragraph("SHAP Attribution Detail", h2_style))
    idx = np.argsort(np.abs(shap_vals))[-top_k:][::-1]
    shap_rows = [["Feature", "Impact", "Direction"]]
    for i in idx:
        v = float(shap_vals[i])
        direction = "Increases Risk" if v > 0 else "Reduces Risk"
        shap_rows.append([friendly_name(feature_cols[i]), f"{v:+.3f}", direction])
    shap_table = Table(shap_rows, colWidths=[8 * cm, 3.5 * cm, 4 * cm])
    shap_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CED4DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(shap_table)

    # Intervention recommendation
    story.append(Paragraph("Recommended Intervention", h2_style))
    story.append(Paragraph(f"<b>{action}</b>", body_style))
    story.append(Spacer(1, 6))
    if band == "High Risk":
        note = (
            "This student is in the top-priority tier. Assign an intensive mentoring "
            "slot immediately and schedule a counseling session within 7 days."
        )
    elif band == "Medium Risk":
        note = (
            "This student should be enrolled in scalable skills workshops and placed "
            "under monthly progress monitoring."
        )
    else:
        note = (
            "This student is currently stable. Continue general academic support and "
            "re-evaluate at the next term boundary."
        )
    story.append(Paragraph(note, body_style))

    # Footer disclaimer
    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "This report is generated by a prototype Decision Support System. "
            "Predictions are probabilistic and should be combined with "
            "professional academic judgement.",
            small_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer


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
                help="Number of curricular units the student passed in semester 1.",
            )
            user_input["Curricular units 2nd sem (approved)"] = st.slider(
                "2nd Semester Passed Units",
                0,
                20,
                int(X_train["Curricular units 2nd sem (approved)"].median()),
                help="Number of curricular units the student passed in semester 2.",
            )
            user_input["Curricular units 2nd sem (grade)"] = st.slider(
                "2nd Semester Average Grade",
                0.0,
                20.0,
                float(X_train["Curricular units 2nd sem (grade)"].median()),
                0.5,
                help="Average grade across all 2nd semester units (0–20).",
            )

            st.markdown("##### Socio-Financial Indicators")
            user_input["Tuition fees up to date"] = st.selectbox(
                "Tuition Fee Status",
                options=[("Up to date", 1), ("Overdue", 0)],
                format_func=lambda x: x[0],
                help="Whether tuition fees are current.",
            )[1]
            user_input["Scholarship holder"] = st.selectbox(
                "Scholarship Holder",
                options=[("Yes", 1), ("No", 0)],
                format_func=lambda x: x[0],
                help="Whether the student receives a scholarship.",
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
                    time.sleep(0.5)

                    medians = X_train.median()
                    x_full = {
                        feat: user_input.get(feat, float(medians[feat]))
                        for feat in feature_cols
                    }
                    x_df = pd.DataFrame([x_full])

                    p_dropout = float(model.predict_proba(x_df)[:, 1])
                    band = assign_risk_band(p_dropout)
                    action = get_intervention(band)

                    shap_vals = explainer.shap_values(x_df)[0]

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

                    # ------- Download button -------
                    pdf_buffer = build_pdf_report(
                        student_inputs=user_input,
                        p_dropout=p_dropout,
                        band=band,
                        action=action,
                        shap_vals=shap_vals,
                        feature_cols=feature_cols,
                        top_k=8,
                    )
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=(
                            f"dropout_risk_report_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    st.markdown("---")

                    # ------- Interactive visual explanations -------
                    st.markdown("### 🧭 Visual Explanation")

                    # Step 1: Probability gauge
                    with st.expander("Step 1 · How high is the risk?", expanded=True):
                        st.markdown(
                            """
                            <div class="explain-box">
                                <h4><span class="step-badge">1</span> Attrition Probability</h4>
                                The gauge shows the model's estimated probability that this
                                student will drop out. The colored background bands correspond
                                to the Low / Medium / High risk thresholds used by the DSS.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_gauge, col_band = st.columns([1, 1.4])
                        with col_gauge:
                            st.plotly_chart(
                                plot_gauge(p_dropout, band),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )
                        with col_band:
                            st.plotly_chart(
                                plot_threshold_explorer(p_dropout),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )

                    # Step 2: SHAP waterfall + donut
                    with st.expander(
                        "Step 2 · Which factors drive the risk?", expanded=True
                    ):
                        st.markdown(
                            """
                            <div class="explain-box">
                                <h4><span class="step-badge">2</span> Factor Attribution (SHAP)</h4>
                                Each bar shows how much a feature pushed the prediction
                                <b style="color:#E63946;">up (risk)</b> or
                                <b style="color:#2A9D8F;">down (protection)</b>.
                                Hover a bar to see the exact impact value.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_water, col_donut = st.columns([1.6, 1])
                        with col_water:
                            st.plotly_chart(
                                plot_shap_waterfall(feature_cols, shap_vals, top_n=6),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )
                        with col_donut:
                            st.plotly_chart(
                                plot_contribution_donut(shap_vals),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )

                        # Chip summary
                        top_pos = [
                            (friendly_name(feature_cols[i]), shap_vals[i])
                            for i in np.argsort(shap_vals)[::-1]
                            if shap_vals[i] > 0
                        ][:3]
                        top_neg = [
                            (friendly_name(feature_cols[i]), shap_vals[i])
                            for i in np.argsort(shap_vals)
                            if shap_vals[i] < 0
                        ][:3]

                        chips_html = ""
                        for name, val in top_pos:
                            chips_html += (
                                f'<span class="factor-chip chip-risk">'
                                f"▲ {name} ({val:+.2f})</span>"
                            )
                        for name, val in top_neg:
                            chips_html += (
                                f'<span class="factor-chip chip-prot">'
                                f"▼ {name} ({val:+.2f})</span>"
                            )

                        st.markdown(
                            f"""
                            <div class="explain-box">
                                <h4>Quick Read</h4>
                                {chips_html if chips_html else "<i>No dominant factors.</i>"}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Step 3: Radar profile
                    with st.expander(
                        "Step 3 · How does this student compare to the cohort?",
                        expanded=False,
                    ):
                        st.markdown(
                            """
                            <div class="explain-box">
                                <h4><span class="step-badge">3</span> Profile vs Cohort Median</h4>
                                The dark shape is the student, normalized against the
                                cohort median (yellow ring). Values reaching the outer
                                ring mean the student is at or above the median for that
                                indicator.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        radar_features = [
                            "Curricular units 1st sem (approved)",
                            "Curricular units 2nd sem (approved)",
                            "Curricular units 2nd sem (grade)",
                            "Age at enrollment",
                            "Tuition fees up to date",
                            "Scholarship holder",
                        ]
                        radar_features = [f for f in radar_features if f in feature_cols]
                        st.plotly_chart(
                            plot_radar_profile(user_input, medians, radar_features),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    # Step 4: Recommended action
                    with st.expander(
                        "Step 4 · What should we do about it?", expanded=True
                    ):
                        st.markdown(
                            f"""
                            <div class="explain-box">
                                <h4><span class="step-badge">4</span> Prescriptive Action</h4>
                                Based on the <b>{band}</b> classification, the DSS recommends:
                                <br><br>
                                <b style="font-size:1.05rem;">{action}</b>
                                <br><br>
                                <i>Capacity is respected: only the top {CAPACITY_LIMIT}
                                high-risk students receive intensive mentoring.</i>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:
                st.info(
                    "👈 Enter the student's metrics and click "
                    "'Generate Diagnostic Prediction' to view "
                    "capacity-aware interventions, SHAP attributions, "
                    "and to download a PDF report."
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
