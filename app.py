# file: app.py
# Run with: streamlit run app.py
# Requires: pip install streamlit xgboost shap plotly scikit-learn pandas numpy reportlab kaleido

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
# PAGE CONFIG MUST BE FIRST
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Dropout Risk DSS",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ==========================================================
# PATHING AND CONSTANTS
# ==========================================================
BASE_DIR = Path(__file__).parent
CSV_NAME = "students_dropout_academic_success.csv"
DATA_FILE_PATH = BASE_DIR / CSV_NAME

LOW_TH = 0.20
HIGH_TH = 0.50
CAPACITY_LIMIT = 200

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
# MODERN CSS
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
    .delta-good { color: #1B4332; font-weight: 700; }
    .delta-bad  { color: #842029; font-weight: 700; }
    .delta-flat { color: #6C757D; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# MODEL AND DATA CACHING
# ==========================================================
@st.cache_resource(show_spinner=True)
def load_and_train_model():
    if not DATA_FILE_PATH.exists():
        st.error(f"Data file not found at: {DATA_FILE_PATH}")
        st.stop()

    df = pd.read_csv(DATA_FILE_PATH)
    df = df[df["target"] != "Enrolled"].copy()
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0})

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
        return "Intensive Mentoring and Counseling"
    elif band == "Medium Risk":
        return "Skills Workshops and Progress Monitoring"
    else:
        return "General Academic Support"


def friendly_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# ==========================================================
# INTERACTIVE PLOTLY COMPONENTS
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
    idx = np.argsort(np.abs(shap_values))[-top_n:]
    names = [friendly_name(feature_names[i]) for i in idx]
    vals = shap_values[idx]
    colors_list = ["#E63946" if v > 0 else "#2A9D8F" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors_list,
            text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top Factors Driving Risk (SHAP Attributions)",
        xaxis_title="Protects (green)  |  Increases Risk (red)",
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
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[LOW_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(42, 157, 143, 0.55)",
            name="Low Risk (< 0.20)",
            hovertemplate="Low Risk<br>0.00 - 0.20<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[HIGH_TH - LOW_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(233, 196, 106, 0.6)",
            name="Medium Risk (0.20 - 0.50)",
            hovertemplate="Medium Risk<br>0.20 - 0.50<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[1.0 - HIGH_TH],
            y=["Risk Band"],
            orientation="h",
            marker_color="rgba(230, 57, 70, 0.6)",
            name="High Risk (>= 0.50)",
            hovertemplate="High Risk<br>0.50 - 1.00<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[probability],
            y=["Risk Band"],
            mode="markers+text",
            marker=dict(
                size=22,
                color="#1D3557",
                symbol="line-ns",
                line=dict(width=3, color="white"),
            ),
            text=[f"  This student: {probability:.1%}"],
            textposition="top center",
            textfont=dict(size=13, color="#1D3557"),
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
    labels = []
    student_vals = []
    cohort_vals = []

    for feat in top_features:
        val = float(user_input.get(feat, medians.get(feat, 0.0)))
        med = float(medians.get(feat, 0.0))
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


def plot_counterfactual_curve(base_input, feature, current_value, model, medians, feature_cols, band_colors=True):
    """Sweep a single feature and plot the probability curve."""
    # Build a sensible sweep range
    if feature in ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"):
        sweep = np.arange(0, 21, 1)
    elif "grade" in feature.lower():
        sweep = np.arange(0, 20.5, 0.5)
    elif feature in ("Age at enrollment",):
        sweep = np.arange(17, 66, 1)
    elif feature in ("Tuition fees up to date", "Scholarship holder", "Gender", "Debtor"):
        sweep = np.array([0, 1])
    else:
        med = float(medians.get(feature, 0.0))
        sweep = np.linspace(max(0, med - 5), med + 5, 25)

    probs = []
    for v in sweep:
        row = dict(base_input)
        row[feature] = float(v)
        x = pd.DataFrame([{f: row.get(f, float(medians[f])) for f in feature_cols}])
        probs.append(float(model.predict_proba(x)[:, 1]))
    probs = np.array(probs)

    fig = go.Figure()

    # Background bands
    fig.add_hrect(y0=0, y1=LOW_TH, fillcolor="rgba(42, 157, 143, 0.10)", line_width=0)
    fig.add_hrect(y0=LOW_TH, y1=HIGH_TH, fillcolor="rgba(233, 196, 106, 0.12)", line_width=0)
    fig.add_hrect(y0=HIGH_TH, y1=1.0, fillcolor="rgba(230, 57, 70, 0.10)", line_width=0)

    fig.add_trace(
        go.Scatter(
            x=sweep,
            y=probs,
            mode="lines",
            line=dict(color="#1D3557", width=3),
            name="Predicted probability",
            hovertemplate=f"{friendly_name(feature)}: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>",
        )
    )

    # Current marker
    cur_prob = float(
        model.predict_proba(
            pd.DataFrame(
                [{f: base_input.get(f, float(medians[f])) for f in feature_cols}]
            )
        )[:, 1]
    )
    fig.add_trace(
        go.Scatter(
            x=[current_value],
            y=[cur_prob],
            mode="markers",
            marker=dict(size=16, color="#E63946", line=dict(width=2, color="white")),
            name="Current value",
            hovertemplate=f"Current: {current_value}<br>Probability: {cur_prob:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Counterfactual Sweep: {friendly_name(feature)}",
        xaxis_title=friendly_name(feature),
        yaxis_title="Dropout Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def plot_capacity_curve(probs_sorted):
    """Lorenz-style curve: cumulative risk captured vs number intervened."""
    n = len(probs_sorted)
    k = np.arange(1, n + 1)
    cumulative_captured = np.cumsum(probs_sorted)
    total_risk = cumulative_captured[-1]
    y = cumulative_captured / total_risk if total_risk > 0 else np.zeros_like(k)

    fig = go.Figure()

    # Diagonal baseline (random selection)
    fig.add_trace(
        go.Scatter(
            x=[0, n],
            y=[0, 1],
            mode="lines",
            line=dict(color="#ADB5BD", width=2, dash="dash"),
            name="Random selection baseline",
            hoverinfo="skip",
        )
    )

    # Model curve
    fig.add_trace(
        go.Scatter(
            x=k,
            y=y,
            mode="lines",
            line=dict(color="#1D3557", width=3),
            name="Model-ranked selection",
            hovertemplate="Intervened: %{x}<br>Risk captured: %{y:.1%}<extra></extra>",
        )
    )

    # Capacity cutoff
    cap = min(CAPACITY_LIMIT, n)
    y_cap = y[cap - 1] if cap > 0 else 0.0

    fig.add_vline(
        x=cap,
        line=dict(color="#E63946", width=2, dash="dash"),
        annotation_text=f"Capacity C={CAPACITY_LIMIT}",
        annotation_position="top",
        annotation_font_color="#E63946",
    )
    fig.add_trace(
        go.Scatter(
            x=[cap],
            y=[y_cap],
            mode="markers+text",
            marker=dict(size=14, color="#E63946", line=dict(width=2, color="white")),
            text=[f"Risk captured at capacity: {y_cap:.1%}"],
            textposition="bottom right",
            textfont=dict(size=12, color="#E63946"),
            name="At capacity",
            hovertemplate=f"Top {cap} students<br>Risk captured: {y_cap:.1%}<extra></extra>",
        )
    )

    # Random baseline at capacity
    random_at_cap = cap / n if n > 0 else 0.0
    fig.add_trace(
        go.Scatter(
            x=[cap],
            y=[random_at_cap],
            mode="markers+text",
            marker=dict(size=12, color="#6C757D", line=dict(width=2, color="white")),
            text=[f"Random at capacity: {random_at_cap:.1%}"],
            textposition="top left",
            textfont=dict(size=11, color="#6C757D"),
            name="Random at capacity",
            hovertemplate=f"Top {cap} by chance<br>Risk captured: {random_at_cap:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Capacity Constraint: Cumulative Risk Captured vs Students Intervened",
        xaxis_title="Number of Students Intervened (ranked by predicted risk)",
        yaxis_title="Fraction of Total Cohort Risk Captured",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=440,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_batch_band_distribution(bands):
    counts = pd.Series(bands).value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"], fill_value=0
    )
    color_map = {
        "Low Risk": "#2A9D8F",
        "Medium Risk": "#E9C46A",
        "High Risk": "#E63946",
    }
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[color_map[b] for b in counts.index],
            text=counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Risk Band Distribution Across Uploaded Cohort",
        xaxis_title="",
        yaxis_title="Number of Students",
        height=340,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==========================================================
# PDF REPORT GENERATION (single-student)
# ==========================================================
def _fig_to_png_bytes(fig, width=700, height=400):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def build_pdf_report(student_inputs, p_dropout, band, action, shap_vals, feature_cols, top_k=8):
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
    story.append(Paragraph("Student Dropout Risk Report", title_style))
    story.append(
        Paragraph(
            f"Capacity-Aware Decision Support System - Generated "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            small_style,
        )
    )
    story.append(Spacer(1, 12))

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
# PDF REPORT GENERATION (batch)
# ==========================================================
def build_batch_pdf_report(summary_df, capacity_df, curve_png, dist_png, model_name="XGBoost"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Batch Cohort Risk Report",
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

    story = []
    story.append(Paragraph("Batch Cohort Risk Report", title_style))
    story.append(
        Paragraph(
            f"Model: {model_name} - Generated "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            small_style,
        )
    )
    story.append(Spacer(1, 12))

    # Summary table
    story.append(Paragraph("Cohort Summary", h2_style))
    summary_rows = [["Metric", "Value"]] + [
        [str(k), str(v)] for k, v in summary_df.items()
    ]
    summary_table = Table(summary_rows, colWidths=[8 * cm, 8 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CED4DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)

    # Capacity curve
    if curve_png:
        story.append(Paragraph("Capacity Constraint Curve", h2_style))
        story.append(RLImage(io.BytesIO(curve_png), width=16 * cm, height=10 * cm))

    # Band distribution
    if dist_png:
        story.append(Paragraph("Risk Band Distribution", h2_style))
        story.append(RLImage(io.BytesIO(dist_png), width=16 * cm, height=8 * cm))

    # Top 25 at-risk students
    story.append(Paragraph("Top 25 Highest-Risk Students", h2_style))
    top_rows = [["Rank", "Student Index", "Probability", "Band", "Action"]]
    for rank, (_, row) in enumerate(capacity_df.head(25).iterrows(), start=1):
        top_rows.append(
            [
                str(rank),
                str(row.get("student_index", "")),
                f"{row['probability']:.2%}",
                str(row["band"]),
                str(row["action"]),
            ]
        )
    top_table = Table(top_rows, colWidths=[1.2 * cm, 2.8 * cm, 2.5 * cm, 2.8 * cm, 6.7 * cm])
    top_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E63946")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF5F5")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CED4DA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(top_table)

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
# COUNTERFACTUAL HELPERS
# ==========================================================
def compute_counterfactual(base_input, feature, new_value, model, medians, feature_cols):
    row = dict(base_input)
    row[feature] = float(new_value)
    x = pd.DataFrame([{f: row.get(f, float(medians[f])) for f in feature_cols}])
    return float(model.predict_proba(x)[:, 1])


def band_change_summary(base_prob, new_prob):
    base_band = assign_risk_band(base_prob)
    new_band = assign_risk_band(new_prob)
    delta = new_prob - base_prob
    if abs(delta) < 0.005:
        cls = "delta-flat"
        arrow = "(unchanged)"
    elif delta < 0:
        cls = "delta-good"
        arrow = "(reduced)"
    else:
        cls = "delta-bad"
        arrow = "(increased)"
    band_note = ""
    if base_band != new_band:
        band_note = f" Band shifts from {base_band} to {new_band}."
    return delta, cls, arrow, band_note


# ==========================================================
# MAIN APP
# ==========================================================
def main():
    model, explainer, feature_cols, X_train = load_and_train_model()
    medians = X_train.median()

    st.markdown(
        '<div class="main-header">Capacity-Aware Decision Support System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Interpretable Student Retention Prototype</div>',
        unsafe_allow_html=True,
    )

    tab_eval, tab_batch, tab_info, tab_fairness = st.tabs(
        [
            "Student Risk Assessment",
            "Batch Cohort Scoring",
            "System Architecture",
            "Fairness Mitigation Audit",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: EVALUATION
    # ---------------------------------------------------------
    with tab_eval:
        col_input, col_results = st.columns([1.2, 2])

        with col_input:
            st.markdown("---")
            st.markdown("### Student Profile")

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
                help="Average grade across all 2nd semester units (0-20).",
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
                "Generate Diagnostic Prediction", use_container_width=True
            )
            st.markdown("---")

        with col_results:
            if evaluate:
                with st.spinner("Executing XGBoost and computing TreeSHAP matrices..."):
                    time.sleep(0.4)

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
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=(
                            f"dropout_risk_report_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    st.markdown("---")
                    st.markdown("### Visual Explanation")

                    with st.expander("Step 1 - How high is the risk?", expanded=True):
                        st.markdown(
                            """
                            <div class="explain-box">
                                <h4><span class="step-badge">1</span> Attrition Probability</h4>
                                The gauge shows the model's estimated probability that this
                                student will drop out. The colored background bands correspond
                                to the Low, Medium, and High risk thresholds used by the DSS.
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

                    with st.expander("Step 2 - Which factors drive the risk?", expanded=True):
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
                                f"{name} ({val:+.2f})</span>"
                            )
                        for name, val in top_neg:
                            chips_html += (
                                f'<span class="factor-chip chip-prot">'
                                f"{name} ({val:+.2f})</span>"
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

                    with st.expander(
                        "Step 3 - How does this student compare to the cohort?",
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

                    with st.expander("Step 4 - What should we do about it?", expanded=True):
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

                    # ---------------- COUNTERFACTUAL PANEL ----------------
                    with st.expander(
                        "Step 5 - What-if: Counterfactual Explorer", expanded=False
                    ):
                        st.markdown(
                            """
                            <div class="explain-box">
                                <h4><span class="step-badge">5</span> Counterfactual What-If Analysis</h4>
                                Move a lever below to simulate a change in the student's profile
                                and immediately see how the predicted risk responds. This turns
                                the DSS from descriptive to actionable: advisors can identify
                                which changes would move the student into a lower risk band.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Feature selector limited to actionable levers
                        actionable_features = [
                            "Curricular units 1st sem (approved)",
                            "Curricular units 2nd sem (approved)",
                            "Curricular units 2nd sem (grade)",
                            "Tuition fees up to date",
                            "Scholarship holder",
                            "Age at enrollment",
                        ]
                        actionable_features = [f for f in actionable_features if f in feature_cols]

                        cf_feature = st.selectbox(
                            "Select a feature to perturb",
                            options=actionable_features,
                            format_func=friendly_name,
                            key="cf_feature",
                        )

                        # Slider range depends on feature type
                        if cf_feature in (
                            "Curricular units 1st sem (approved)",
                            "Curricular units 2nd sem (approved)",
                        ):
                            cf_value = st.slider(
                                f"Simulated value for {friendly_name(cf_feature)}",
                                min_value=0,
                                max_value=20,
                                value=int(user_input.get(cf_feature, 0)),
                                step=1,
                                key="cf_value_int",
                            )
                        elif "grade" in cf_feature.lower():
                            cf_value = st.slider(
                                f"Simulated value for {friendly_name(cf_feature)}",
                                min_value=0.0,
                                max_value=20.0,
                                value=float(user_input.get(cf_feature, 10.0)),
                                step=0.5,
                                key="cf_value_float",
                            )
                        elif cf_feature == "Age at enrollment":
                            cf_value = st.slider(
                                f"Simulated value for {friendly_name(cf_feature)}",
                                min_value=17,
                                max_value=65,
                                value=int(user_input.get(cf_feature, 20)),
                                step=1,
                                key="cf_value_age",
                            )
                        else:
                            cf_value = st.selectbox(
                                f"Simulated value for {friendly_name(cf_feature)}",
                                options=[("No", 0), ("Yes", 1)] if cf_feature != "Scholarship holder" else [("Yes", 1), ("No", 0)],
                                format_func=lambda x: x[0],
                                key="cf_value_bin",
                            )[1]

                        # Compute counterfactual
                        cf_prob = compute_counterfactual(
                            user_input, cf_feature, cf_value, model, medians, feature_cols
                        )
                        delta, delta_cls, arrow, band_note = band_change_summary(
                            p_dropout, cf_prob
                        )

                        col_cf1, col_cf2, col_cf3 = st.columns(3)
                        col_cf1.metric(
                            "Baseline Probability",
                            f"{p_dropout:.1%}",
                            help="Current predicted probability.",
                        )
                        col_cf2.metric(
                            "Simulated Probability",
                            f"{cf_prob:.1%}",
                            delta=f"{delta:+.2%}",
                            help="Predicted probability under the counterfactual.",
                        )
                        col_cf3.metric(
                            "Risk Band Shift",
                            assign_risk_band(cf_prob),
                            help="Band under the counterfactual.",
                        )

                        st.markdown(
                            f"""
                            <div class="explain-box">
                                Moving <b>{friendly_name(cf_feature)}</b>
                                from <b>{user_input.get(cf_feature, 'current')}</b>
                                to <b>{cf_value}</b> would change the predicted dropout
                                probability by <span class="{delta_cls}">{delta:+.2%}</span> {arrow}.
                                {band_note}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Sweep curve
                        current_val = user_input.get(cf_feature, cf_value)
                        sweep_fig = plot_counterfactual_curve(
                            user_input, cf_feature, current_val, model, medians, feature_cols
                        )
                        st.plotly_chart(
                            sweep_fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    with st.expander(
                        "Step 6 - Recommended action summary", expanded=False
                    ):
                        st.markdown(
                            f"""
                            <div class="explain-box">
                                <h4><span class="step-badge">6</span> Final Recommendation</h4>
                                <b>{action}</b>
                                <br><br>
                                The counterfactual explorer above can identify one or two
                                high-leverage indicators to discuss with the student. Target
                                those indicators first before escalating to intensive support.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:
                st.info(
                    "Enter the student's metrics and click "
                    "'Generate Diagnostic Prediction' to view "
                    "capacity-aware interventions, SHAP attributions, "
                    "counterfactual analysis, and to download a PDF report."
                )

    # ---------------------------------------------------------
    # TAB 2: BATCH COHORT SCORING
    # ---------------------------------------------------------
    with tab_batch:
        st.markdown("### Batch Cohort Scoring")
        st.markdown(
            """
            Upload a CSV with the same schema as the training data (excluding the
            `target` column) to score an entire cohort in one pass. The DSS will
            apply the capacity constraint (top 200 students) automatically and
            return an annotated CSV plus a summary PDF.
            """
        )

        st.markdown(
            """
            <div class="explain-box">
                <h4>Expected CSV Format</h4>
                The file should contain one row per student. Columns must match the
                model's feature names. Missing optional columns are imputed with
                cohort medians. A helper template can be generated below.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload cohort CSV",
                type=["csv"],
                help="One row per student. Columns must match model features.",
            )
        with col_up2:
            template_df = pd.DataFrame(
                [{f: float(X_train[f].median()) for f in feature_cols}]
            )
            st.download_button(
                label="Download CSV Template",
                data=template_df.to_csv(index=False).encode("utf-8"),
                file_name="cohort_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()

            st.markdown("#### Uploaded Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            st.caption(f"Total rows uploaded: {len(batch_df)}")

            if st.button("Run Batch Scoring", use_container_width=True):
                with st.spinner("Scoring cohort and applying capacity constraint..."):
                    time.sleep(0.3)

                    # Align columns: use uploaded features where present, impute the rest
                    working = batch_df.copy()
                    for f in feature_cols:
                        if f not in working.columns:
                            working[f] = float(medians[f])
                        else:
                            working[f] = pd.to_numeric(working[f], errors="coerce")
                            working[f] = working[f].fillna(float(medians[f]))

                    X_batch = working[feature_cols]
                    probs = model.predict_proba(X_batch)[:, 1]

                    scored = working.copy()
                    scored["probability"] = probs
                    scored["band"] = [assign_risk_band(p) for p in probs]

                    # Rank by probability descending; apply capacity to high-risk
                    scored = scored.sort_values("probability", ascending=False).reset_index(drop=True)
                    scored["student_index"] = np.arange(len(scored))

                    actions = []
                    intensive_allocated = 0
                    for _, row in scored.iterrows():
                        if row["band"] == "High Risk" and intensive_allocated < CAPACITY_LIMIT:
                            actions.append("Intensive Mentoring and Counseling")
                            intensive_allocated += 1
                        elif row["band"] == "High Risk":
                            actions.append("Skills Workshops and Progress Monitoring (capacity overflow)")
                        elif row["band"] == "Medium Risk":
                            actions.append("Skills Workshops and Progress Monitoring")
                        else:
                            actions.append("General Academic Support")
                    scored["action"] = actions

                    # ---- Summary ----
                    n_total = len(scored)
                    n_high = int((scored["band"] == "High Risk").sum())
                    n_med = int((scored["band"] == "Medium Risk").sum())
                    n_low = int((scored["band"] == "Low Risk").sum())
                    n_intensive = int((scored["action"] == "Intensive Mentoring and Counseling").sum())
                    n_overflow = int(
                        (scored["action"] == "Skills Workshops and Progress Monitoring (capacity overflow)").sum()
                    )
                    mean_prob = float(scored["probability"].mean())

                    probs_sorted = np.sort(probs)[::-1]
                    total_risk = probs_sorted.sum()
                    cap = min(CAPACITY_LIMIT, n_total)
                    risk_at_cap = probs_sorted[:cap].sum() / total_risk if total_risk > 0 else 0.0
                    random_at_cap = cap / n_total if n_total > 0 else 0.0
                    lift = (risk_at_cap / random_at_cap) if random_at_cap > 0 else 0.0

                    st.markdown("---")
                    st.markdown("### Cohort Summary")

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Students Scored", f"{n_total}")
                    col_s2.metric("High Risk", f"{n_high}")
                    col_s3.metric("Medium Risk", f"{n_med}")
                    col_s4.metric("Low Risk", f"{n_low}")

                    col_s5, col_s6, col_s7, col_s8 = st.columns(4)
                    col_s5.metric("Mean Probability", f"{mean_prob:.1%}")
                    col_s6.metric("Intensive Slots Used", f"{n_intensive} / {CAPACITY_LIMIT}")
                    col_s7.metric("Capacity Overflow", f"{n_overflow}")
                    col_s8.metric(
                        "Risk Captured at Capacity",
                        f"{risk_at_cap:.1%}",
                        delta=f"{lift:.2f}x lift vs random",
                    )

                    # ---- Capacity curve ----
                    st.markdown("---")
                    st.markdown("### Capacity Constraint Visualization")
                    st.markdown(
                        """
                        <div class="explain-box">
                            <h4>Interpreting the Capacity Curve</h4>
                            The curve shows how much of the cohort's total predicted
                            dropout risk is captured as we intervene on more students,
                            ranked by the model. The vertical red line marks the
                            operational capacity C = 200. A steep initial climb means
                            the model concentrates risk in a small number of students,
                            which is exactly what a capacity-limited DSS needs.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    cap_fig = plot_capacity_curve(probs_sorted)
                    st.plotly_chart(cap_fig, use_container_width=True, config={"displayModeBar": False})

                    # ---- Band distribution ----
                    st.markdown("---")
                    st.markdown("### Risk Band Distribution")
                    dist_fig = plot_batch_band_distribution(scored["band"].tolist())
                    st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})

                    # ---- Priority table ----
                    st.markdown("---")
                    st.markdown("### Priority Queue (Top 50 Highest-Risk Students)")
                    preview_cols = ["student_index", "probability", "band", "action"]
                    st.dataframe(
                        scored[preview_cols].head(50).style.format({"probability": "{:.2%}"}),
                        use_container_width=True,
                    )

                    # ---- Download annotated CSV ----
                    csv_bytes = scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Annotated Cohort CSV",
                        data=csv_bytes,
                        file_name=(
                            f"cohort_scored_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # ---- Download batch PDF ----
                    curve_png = _fig_to_png_bytes(cap_fig, width=800, height=500)
                    dist_png = _fig_to_png_bytes(dist_fig, width=700, height=400)

                    summary_df = pd.Series(
                        {
                            "Students scored": n_total,
                            "High risk": n_high,
                            "Medium risk": n_med,
                            "Low risk": n_low,
                            "Mean probability": f"{mean_prob:.2%}",
                            "Intensive slots used": f"{n_intensive} / {CAPACITY_LIMIT}",
                            "Capacity overflow": n_overflow,
                            "Risk captured at capacity": f"{risk_at_cap:.2%}",
                            "Lift vs random": f"{lift:.2f}x",
                        }
                    )

                    batch_pdf = build_batch_pdf_report(
                        summary_df=summary_df,
                        capacity_df=scored[["student_index", "probability", "band", "action"]],
                        curve_png=curve_png,
                        dist_png=dist_png,
                    )
                    st.download_button(
                        label="Download Batch PDF Report",
                        data=batch_pdf,
                        file_name=(
                            f"cohort_report_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )

    # ---------------------------------------------------------
    # TAB 3: SYSTEM ARCHITECTURE
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
            max sum (p_i * I(p_i >= T_H)) * a_i    s.t.    sum a_i <= C
            ```

            This explicitly forces the algorithm to prioritize the top 200
            high-risk cases for Intensive Mentoring, dynamically routing
            capacity overflow to scalable workshops. The Batch Cohort Scoring
            tab visualizes this constraint directly through the cumulative
            risk-captured curve.
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # TAB 4: FAIRNESS MITIGATION
    # ---------------------------------------------------------
    with tab_fairness:
        st.markdown(
            """
            ### Active Equal Opportunity Mitigation

            Rather than passively auditing bias, the framework actively
            enforces Equal Opportunity by optimizing group-specific
            probability thresholds (T_H,g). This neutralizes historical
            demographic disparities in intervention allocation.
            """
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
                * **Sensitivity Gap (Delta TPR): 0.022**
                """
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
                * **Sensitivity Gap (Delta TPR): 0.003**
                """
            )


if __name__ == "__main__":
    main()
