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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.platypus.flowables import HRFlowable

# ==========================================================
# PAGE CONFIG MUST BE FIRST
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Dropout Risk DSS",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

# ==========================================================
# WHITE-LABEL & I18N CONFIGURATION
# ==========================================================
WHITE_LABEL = {
    "institution_name": "Global University",
    "logo_svg": '''
<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 9 12 4 2 9l10 5 10-5z" /><path d="M6 11.5V16c0 1.2 2.7 3 6 3s6-1.8 6-3v-4.5" /><path d="M22 9v6" /></svg>
''',
    "contact_email": "support@globaluniversity.edu"
}

CURRENT_LOCALE = "en"
UI_TEXT = {
    "en": {
        "intro_title": "Welcome to the Capacity-Aware DSS",
        "intro_promise": "Actionable, interpretable student retention predictions aligned with your institution's mentoring capacity.",
        "intro_step_1": "1. Enter a student profile or upload a cohort.",
        "intro_step_2": "2. Review risk drivers and simulate counterfactuals.",
        "intro_step_3": "3. Export print-ready, accessible PDF reports.",
        "intro_tour": "Guided Tour (20s): The 'Student Assessment' tab allows deep-dives into single profiles. 'Cohort Scoring' handles CSV batch uploads. 'How It Works' details our XGBoost/SHAP architecture. 'Fairness Audit' verifies equal opportunity metrics.",
        "intro_privacy": "Privacy Note: No student PII is stored server-side. Session history is cleared automatically.",
        "intro_btn": "Start assessment",
        "outro_summary": "Assessment Complete",
        "outro_next": "Review the downloaded PDF report and utilize the action plan as a structured starting point.",
        "outro_btn": "Assess another student",
        "footer_privacy": "FERPA/GDPR Compliant · Data stays in this session.",
    }
}


def _t(key: str) -> str:
    return UI_TEXT.get(CURRENT_LOCALE, UI_TEXT["en"]).get(key, key)


# ==========================================================
# PATHING AND CONSTANTS
# ==========================================================
BASE_DIR = Path(__file__).parent
CSV_NAME = "students_dropout_academic_success.csv"
DATA_FILE_PATH = BASE_DIR / CSV_NAME

LOW_TH = 0.20
HIGH_TH = 0.50
CAPACITY_LIMIT = 200
MODEL_NAME = "XGBoost (gradient-boosted trees)"
MODEL_VERSION = "v2.0 - Decision Support Prototype"

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
# PRODUCT UI THEME (Design System)
# ==========================================================
import plotly.io as pio
import streamlit.components.v1 as components

_FONT = "Inter, 'Segoe UI', Arial, sans-serif"
pio.templates["campus"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=_FONT, size=12.5, color="#0E1B3D"),
        title=dict(font=dict(size=14, color="#0E1B3D", family=_FONT), x=0, xanchor="left"),
        colorway=["#3B5BDB", "#12A594", "#F5A524", "#DC3F4A", "#1B2A5C"],
        xaxis=dict(gridcolor="#E4EAF2", linecolor="#D5DCE8", zerolinecolor="#1B2A5C",
                   ticks="outside", tickcolor="#D5DCE8", title=dict(font=dict(size=11.5))),
        yaxis=dict(gridcolor="#E4EAF2", linecolor="#D5DCE8", zerolinecolor="#1B2A5C",
                   title=dict(font=dict(size=11.5))),
        hoverlabel=dict(bgcolor="#0E1B3D", bordercolor="#0E1B3D",
                        font=dict(color="#FFFFFF", family=_FONT, size=12)),
        legend=dict(font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
    )
)
pio.templates.default = "campus"

st.markdown(
    """
    """,
    unsafe_allow_html=True,
)

# Small JS enhancer for animated numbers and accessibility tweaks
components.html(
    """
    """,
    height=0,
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

    regex = re.compile(r"[$$$$<>]", re.IGNORECASE)
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


def get_intervention_detail(band: str) -> str:
    if band == "High Risk":
        return (
            "Assign an intensive mentoring slot immediately (within 7 days) and "
            "schedule a counseling session. Reassess monthly. This tier is capped "
            f"at {CAPACITY_LIMIT} students university-wide; if the cap is reached, "
            "route the student to workshops and monitoring instead."
        )
    elif band == "Medium Risk":
        return (
            "Enroll the student in scalable skills workshops (study skills, time "
            "management, quantitative foundations). Place under monthly progress "
            "monitoring. Escalate to intensive mentoring if the next assessment "
            "shows deterioration."
        )
    else:
        return (
            "Continue general academic support. Re-evaluate the profile at the next "
            "term boundary. No targeted resource allocation is indicated at this time."
        )


def friendly_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# ==========================================================
# INTERACTIVE PLOTLY COMPONENTS
# ==========================================================
def _chart_label(feature: str) -> str:
    short = {
        "Curricular units 1st sem (approved)": "1st Sem Units",
        "Curricular units 2nd sem (approved)": "2nd Sem Units",
        "Curricular units 2nd sem (grade)": "2nd Sem Grade",
        "Curricular units 1st sem (grade)": "1st Sem Grade",
        "Tuition fees up to date": "Tuition Current",
        "Scholarship holder": "Scholarship",
        "Gender": "Gender",
        "Age at enrollment": "Age",
        "Debtor": "Debtor",
        "Application mode": "Application Mode",
        "Course": "Course",
        "Previous qualification (grade)": "Previous Grade",
        "Admission grade": "Admission Grade",
    }
    return short.get(feature, friendly_name(feature)[:22] + ("…" if len(friendly_name(feature)) > 22 else ""))


def _plot_base(height=320, margin=None):
    fig = go.Figure()
    fig.update_layout(
        autosize=True, height=height, margin=margin or dict(l=46, r=22, t=54, b=48),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=11.5, color="#1B2A5C"),
        title=dict(font=dict(size=13, color="#0E1B3D", family=_FONT), x=0, xanchor="left", y=.98),
        legend=dict(bgcolor="rgba(255,255,255,.88)", bordercolor="#E4EAF2", borderwidth=1, font=dict(size=10.5)),
        hoverlabel=dict(bgcolor="#0E1B3D", font=dict(color="#FFFFFF", family=_FONT, size=11)),
    )
    return fig


def plot_gauge(probability, band):
    color = "#12A594" if band == "Low Risk" else "#F5A524" if band == "Medium Risk" else "#DC3F4A"
    fig = _plot_base(height=270, margin=dict(l=18, r=18, t=32, b=18))
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1B2A5C"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "#FFFFFF", "borderwidth": 1, "bordercolor": "#E4EAF2",
            "steps": [
                {"range": [0, 20], "color": "rgba(18,165,148,.10)"},
                {"range": [20, 50], "color": "rgba(245,165,36,.10)"},
                {"range": [50, 100], "color": "rgba(220,63,74,.10)"},
            ],
            "threshold": {"line": {"color": "#0E1B3D", "width": 2}, "thickness": 0.72, "value": probability * 100},
        },
    ))
    return fig


def plot_shap_waterfall(feature_names, shap_values, top_n=6):
    idx = np.argsort(np.abs(shap_values))[-top_n:][::-1]
    names = [friendly_name(feature_names[i]) for i in idx]
    axis_names = [_chart_label(feature_names[i]) for i in idx]
    vals = shap_values[idx]
    colors_list = ["#DC3F4A" if v > 0 else "#12A594" for v in vals]
    fig = _plot_base(height=355, margin=dict(l=124, r=26, t=36, b=54))
    fig.add_trace(go.Bar(
        x=vals[::-1], y=axis_names[::-1], orientation="h", marker_color=colors_list[::-1],
        customdata=np.array(names[::-1], dtype=object),
        text=[f"{v:+.2f}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
        hovertemplate="**%{customdata}**<br><br>SHAP impact: %{x:+.3f}",
    ))
    fig.update_layout(title="Top risk drivers", showlegend=False, xaxis_title="SHAP impact (log-odds)")
    return fig


def plot_contribution_donut(shap_values):
    pos = float(np.sum(shap_values[shap_values > 0]))
    neg = float(np.sum(np.abs(shap_values[shap_values < 0])))
    fig = _plot_base(height=315, margin=dict(l=18, r=18, t=48, b=18))
    fig.add_trace(go.Pie(
        labels=["Risk-increasing", "Risk-reducing"], values=[pos, neg], hole=0.68,
        marker_colors=["#DC3F4A", "#12A594"], textinfo="percent",
        hovertemplate="**%{label}**<br><br>Magnitude: %{value:.3f}<br><br>%{percent}", sort=False,
    ))
    fig.update_layout(title="Contribution balance", legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top"))
    return fig


def plot_threshold_explorer(probability):
    fig = _plot_base(height=245, margin=dict(l=28, r=24, t=54, b=54))
    colors_band = ["rgba(18,165,148,.72)", "rgba(245,165,36,.78)", "rgba(220,63,74,.76)"]
    names = ["Low", "Medium", "High"]
    widths = [LOW_TH, HIGH_TH - LOW_TH, 1.0 - HIGH_TH]
    starts = np.cumsum([0] + widths[:-1])
    for name, width, start_x, color in zip(names, widths, starts, colors_band):
        fig.add_trace(go.Bar(
            x=[width], y=["Risk band"], orientation="h", base=[start_x],
            marker_color=color, name=name, text=[name], textposition="inside", insidetextanchor="middle",
            hovertemplate=f"{name} Risk<br><br>Range: {start_x:.0%}–{start_x+width:.0%}",
        ))
    fig.add_trace(go.Scatter(
        x=[probability], y=["Risk band"], mode="markers",
        marker=dict(size=17, color="#0E1B3D", line=dict(width=3, color="#FFFFFF")),
        customdata=[f"{probability:.1%}"], name="Student",
        hovertemplate="**This student**<br><br>Probability: %{customdata}",
    ))
    fig.update_layout(barmode="stack", showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(showticklabels=False))
    return fig


def plot_radar_profile(user_input, medians, top_features):
    labels, student_vals, cohort_vals, hover = [], [], [], []
    for feat in top_features:
        val = float(user_input.get(feat, medians.get(feat, 0.0)))
        med = float(medians.get(feat, 0.0))
        denom = med if med > 0 else 1.0
        student_norm = max(0.0, min(1.0, val / denom)) if denom else 0.0
        labels.append(_chart_label(feat))
        student_vals.append(student_norm)
        cohort_vals.append(1.0)
        hover.append(f"{friendly_name(feat)}<br><br>Student: {val:.2f}<br><br>Cohort median: {med:.2f}")
    if labels:
        labels.append(labels[0])
        student_vals.append(student_vals[0])
        cohort_vals.append(cohort_vals[0])
        hover.append(hover[0])

    fig = _plot_base(height=385, margin=dict(l=44, r=44, t=42, b=46))
    fig.add_trace(go.Scatterpolar(r=student_vals, theta=labels, fill="toself", name="Student", line=dict(color="#3B5BDB"), text=hover, hovertemplate="%{text}"))
    fig.add_trace(go.Scatterpolar(r=cohort_vals, theta=labels, fill="toself", name="Cohort median", line=dict(color="#F5A524"), hovertemplate="Median"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.03))
    return fig


def plot_counterfactual_curve(base_input, feature, current_value, model, medians, feature_cols):
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
    cur_row = pd.DataFrame([{f: base_input.get(f, float(medians[f])) for f in feature_cols}])
    cur_prob = float(model.predict_proba(cur_row)[:, 1])

    fig = _plot_base(height=355, margin=dict(l=56, r=26, t=52, b=56))
    fig.add_hrect(y0=0, y1=LOW_TH, fillcolor="rgba(18,165,148,.08)", line_width=0)
    fig.add_hrect(y0=LOW_TH, y1=HIGH_TH, fillcolor="rgba(245,165,36,.09)", line_width=0)
    fig.add_hrect(y0=HIGH_TH, y1=1.0, fillcolor="rgba(220,63,74,.08)", line_width=0)
    fig.add_trace(go.Scatter(x=sweep, y=probs, mode="lines", line=dict(color="#3B5BDB", width=3), name="Predicted probability"))
    fig.add_trace(go.Scatter(x=[current_value], y=[cur_prob], mode="markers", marker=dict(size=15, color="#DC3F4A", line=dict(width=2.5, color="#FFFFFF")), name="Current value"))
    fig.update_layout(title=f"What-if response · {_chart_label(feature)}", yaxis=dict(range=[0, 1], tickformat=".0%"), legend=dict(orientation="h", x=0, y=1.03, yanchor="bottom"))
    return fig


def plot_capacity_curve(probs_sorted):
    n = len(probs_sorted)
    k = np.arange(1, n + 1)
    cumulative_captured = np.cumsum(probs_sorted)
    total_risk = cumulative_captured[-1] if len(cumulative_captured) else 0.0
    y = cumulative_captured / total_risk if total_risk > 0 else np.zeros_like(k, dtype=float)
    cap = min(CAPACITY_LIMIT, n)
    y_cap = y[cap - 1] if cap > 0 else 0.0
    random_at_cap = cap / n if n > 0 else 0.0

    fig = _plot_base(height=455, margin=dict(l=62, r=26, t=74, b=64))
    fig.add_trace(go.Scatter(x=[0, n], y=[0, 1], mode="lines", line=dict(color="#B6C0CF", dash="dash"), name="Random baseline", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=y, mode="lines", line=dict(color="#3B5BDB", width=3), name="Model-ranked"))
    fig.add_vline(x=cap, line=dict(color="#DC3F4A", width=2, dash="dot"))
    if cap > 0:
        fig.add_trace(go.Scatter(x=[cap], y=[y_cap], mode="markers", marker=dict(size=13, color="#DC3F4A"), name="At capacity"))
        fig.add_trace(go.Scatter(x=[cap], y=[random_at_cap], mode="markers", marker=dict(size=11, color="#7C8798"), name="Random at capacity"))
    fig.update_layout(title="Capacity allocation curve", yaxis=dict(range=[0, 1.05], tickformat=".0%"), legend=dict(orientation="h", x=0, y=1.035, yanchor="bottom"))
    return fig


def plot_batch_band_distribution(bands):
    counts = pd.Series(bands).value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    color_map = {"Low Risk": "#12A594", "Medium Risk": "#F5A524", "High Risk": "#DC3F4A"}
    fig = _plot_base(height=335, margin=dict(l=48, r=22, t=54, b=52))
    fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=[color_map[b] for b in counts.index], text=counts.values, textposition="outside"))
    fig.update_layout(title="Risk-band distribution", showlegend=False, yaxis=dict(range=[0, max(1, counts.max()) * 1.18]))
    return fig


def plot_probability_histogram(probs):
    fig = _plot_base(height=335, margin=dict(l=50, r=22, t=54, b=52))
    fig.add_trace(go.Histogram(x=probs, nbinsx=25, marker_color="#3B5BDB"))
    fig.add_vline(x=LOW_TH, line=dict(color="#12A594", width=2, dash="dot"))
    fig.add_vline(x=HIGH_TH, line=dict(color="#DC3F4A", width=2, dash="dot"))
    fig.update_layout(title="Predicted probability distribution", showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"))
    return fig


# ==========================================================
# PDF HELPERS
# ==========================================================
def _fig_to_png_bytes(fig, width=700, height=400):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _styles():
    styles = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("CoverTitle", parent=styles["Title"], textColor=colors.HexColor("#0E1B3D"), fontSize=26, leading=30, spaceAfter=5, fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle("SubTitle", parent=styles["Heading2"], textColor=colors.HexColor("#3B5BDB"), fontSize=11.5, leading=15, spaceAfter=4, fontName="Helvetica-Bold"),
        "h1": ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#0E1B3D"), fontSize=16, leading=19, spaceBefore=7, spaceAfter=8, fontName="Helvetica-Bold"),
        "body": ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.3, leading=13, textColor=colors.HexColor("#1B2A5C")),
        "small": ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#5A6A84")),
        "table": ParagraphStyle("Table", parent=styles["BodyText"], fontSize=7.8, leading=9.6, textColor=colors.HexColor("#1B2A5C")),
        "table_header": ParagraphStyle("TableHeader", parent=styles["BodyText"], fontSize=7.7, leading=9.2, textColor=colors.white, fontName="Helvetica-Bold"),
    }


def _std_table(data, col_widths=None):
    S = _styles()
    wrapped = [[Paragraph(str(cell).replace("\n", "<br/>"), S["table_header"] if r == 0 else S["table"]) for cell in row] for r, row in enumerate(data)]
    t = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E1B3D")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FB")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E4EAF2")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _pdf_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#E4EAF2"))
    canvas.setLineWidth(.5)
    canvas.line(1.75 * cm, 1.25 * cm, w - 1.75 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#5A6A84"))
    canvas.drawString(1.75 * cm, .78 * cm, f"{WHITE_LABEL['institution_name']} - DSS")
    canvas.drawRightString(w - 1.75 * cm, .78 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(student_inputs, p_dropout, band, action, shap_vals, feature_cols, medians, model, counterfactual_feature=None, counterfactual_curve_png=None, radar_png=None, top_k=10):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.75*cm, rightMargin=1.75*cm, topMargin=1.45*cm, bottomMargin=1.6*cm)
    S = _styles(); story = []

    story.append(Paragraph("Student Dropout Risk Report", S["cover_title"]))
    story.append(Paragraph(f"{WHITE_LABEL['institution_name']} - Capacity-Aware DSS", S["subtitle"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", S["small"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Probability and Band Placement", S["h1"]))
    story.append(Paragraph(f"**Probability:** {p_dropout:.1%} | **Risk Band:** {band}", S["body"]))
    story.append(Paragraph(f"**Recommended Action:** {action}", S["body"]))

    gauge_png = _fig_to_png_bytes(plot_gauge(p_dropout, band), width=400, height=200)
    if gauge_png:
        story.append(RLImage(io.BytesIO(gauge_png), width=10*cm, height=5*cm))

    story.append(Paragraph("2. Submitted Profile", S["h1"]))
    profile_rows = [["Indicator", "Submitted", "Cohort median"]]
    for k, v in student_inputs.items():
        profile_rows.append([friendly_name(k), str(v), f"{float(medians.get(k, 0)):.2f}"])
    story.append(_std_table(profile_rows, col_widths=[8*cm, 4*cm, 4*cm]))

    if radar_png:
        story.append(Spacer(1, 10))
        story.append(RLImage(io.BytesIO(radar_png), width=12.6*cm, height=7.25*cm))

    story.append(PageBreak())
    story.append(Paragraph("3. Risk Drivers (SHAP)", S["h1"]))
    idx = np.argsort(np.abs(shap_vals))[-min(top_k, 8):][::-1]
    shap_rows = [["Feature", "SHAP impact", "Direction"]]
    for i in idx:
        v = float(shap_vals[i])
        shap_rows.append([friendly_name(feature_cols[i]), f"{v:+.3f}", "Increases risk" if v > 0 else "Reduces risk"])
    story.append(_std_table(shap_rows, col_widths=[8*cm, 4*cm, 4*cm]))

    if counterfactual_curve_png:
        story.append(Paragraph("4. What-If Scenario", S["h1"]))
        story.append(RLImage(io.BytesIO(counterfactual_curve_png), width=16.3*cm, height=7.05*cm))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=.65, color=colors.HexColor("#E4EAF2")))
    story.append(Paragraph("Disclaimer: Model outputs are probabilistic and intended for decision support, not fully automated decisions.", S["small"]))

    doc.build(story, onFirstPage=_pdf_page, onLaterPages=_pdf_page)
    buffer.seek(0)
    return buffer


def build_batch_pdf_report(scored_df, summary_df, curve_png=None, dist_png=None, hist_png=None, top_n=50):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.75*cm, rightMargin=1.75*cm, topMargin=1.45*cm, bottomMargin=1.6*cm)
    S = _styles(); story = []

    story.append(Paragraph("Batch Cohort Risk Report", S["cover_title"]))
    story.append(Paragraph(f"{WHITE_LABEL['institution_name']} - DSS", S["subtitle"]))
    story.append(Spacer(1, 10))

    snap = [["Metric", "Value"]] + [[str(k), str(v)] for k, v in summary_df.items()]
    story.append(_std_table(snap, col_widths=[8*cm, 8*cm]))

    if curve_png:
        story.append(Paragraph("Capacity Allocation Curve", S["h1"]))
        story.append(RLImage(io.BytesIO(curve_png), width=16.3*cm, height=8.75*cm))

    story.append(PageBreak())
    story.append(Paragraph("Priority Queue (Top Cases)", S["h1"]))
    top_rows = [["Student idx", "Probability", "Band", "Action"]]
    for _, row in scored_df.head(top_n).iterrows():
        top_rows.append([str(row.get("student_index", "")), f"{float(row['probability']):.1%}", str(row["band"]), str(row["action"])])
    story.append(_std_table(top_rows, col_widths=[2.5*cm, 2.5*cm, 3*cm, 8*cm]))

    doc.build(story, onFirstPage=_pdf_page, onLaterPages=_pdf_page)
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
        cls, arrow = "delta-flat", "(unchanged)"
    elif delta < 0:
        cls, arrow = "delta-good", "(reduced)"
    else:
        cls, arrow = "delta-bad", "(increased)"
    band_note = f" Band shifts from {base_band} to {new_band}." if base_band != new_band else ""
    return delta, cls, arrow, band_note


# ==========================================================
# APP VIEWS & RENDERERS
# ==========================================================
def render_intro_panel():
    if not st.session_state.get("intro_dismissed", False):
        st.markdown(
            f"""
            <div class="intro-panel">

            ## {_t('intro_title')}

            {_t('intro_promise')}

            **{_t('intro_step_1')}**

            **{_t('intro_step_2')}**

            **{_t('intro_step_3')}**

            *{_t('intro_tour')}*

            {_t('intro_privacy')}

            </div>
            """, unsafe_allow_html=True
        )
        if st.button(_t("intro_btn"), type="primary"):
            st.session_state["intro_dismissed"] = True
            st.rerun()


def render_outro_panel():
    st.markdown("---")
    st.markdown(
        f"""
        <div class="outro-panel">

        ### {_t('outro_summary')}

        {_t('outro_next')}

        </div>
        """, unsafe_allow_html=True
    )
    if st.button(_t("outro_btn")):
        if "assessment" in st.session_state:
            del st.session_state["assessment"]
        st.rerun()
    st.markdown(
        f"""
        <div class="app-footer">

        {WHITE_LABEL['institution_name']} • {_t('footer_privacy')}
        Version {MODEL_VERSION} • [Support](mailto:{WHITE_LABEL['contact_email']})

        </div>
        """, unsafe_allow_html=True
    )


def run_batch(batch_df, model, medians, feature_cols):
    working = batch_df.copy()
    for f in feature_cols:
        if f not in working.columns:
            working[f] = float(medians[f])
        else:
            working[f] = pd.to_numeric(working[f], errors="coerce").fillna(float(medians[f]))

    X_batch = working[feature_cols]
    probs = model.predict_proba(X_batch)[:, 1]

    scored = working.copy()
    scored["probability"] = probs
    scored["band"] = [assign_risk_band(p) for p in probs]
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

    n_total = len(scored)
    n_high = int((scored["band"] == "High Risk").sum())
    n_med = int((scored["band"] == "Medium Risk").sum())
    n_low = int((scored["band"] == "Low Risk").sum())
    n_intensive = int((scored["action"] == "Intensive Mentoring and Counseling").sum())
    n_overflow = int((scored["action"].str.contains("capacity overflow", na=False)).sum())
    mean_prob = float(scored["probability"].mean())
    median_prob = float(scored["probability"].median())

    probs_sorted = np.sort(probs)[::-1]
    total_risk = probs_sorted.sum()
    cap = min(CAPACITY_LIMIT, n_total)
    risk_at_cap = probs_sorted[:cap].sum() / total_risk if total_risk > 0 else 0.0
    random_at_cap = cap / n_total if n_total > 0 else 0.0
    lift = (risk_at_cap / random_at_cap) if random_at_cap > 0 else 0.0

    cap_fig = plot_capacity_curve(probs_sorted)
    dist_fig = plot_batch_band_distribution(scored["band"].tolist())
    hist_fig = plot_probability_histogram(probs)

    summary_df = pd.Series({
        "Students scored": n_total,
        "High risk": f"{n_high} ({n_high / n_total:.1%})" if n_total else "0",
        "Medium risk": f"{n_med} ({n_med / n_total:.1%})" if n_total else "0",
        "Low risk": f"{n_low} ({n_low / n_total:.1%})" if n_total else "0",
        "Mean probability": f"{mean_prob:.2%}",
        "Intensive slots used": f"{n_intensive} / {CAPACITY_LIMIT}",
        "Capacity overflow": n_overflow,
        "Risk captured at capacity": f"{risk_at_cap:.2%}",
    })

    batch_pdf = build_batch_pdf_report(
        scored_df=scored[["student_index", "probability", "band", "action"]],
        summary_df=summary_df,
        curve_png=_fig_to_png_bytes(cap_fig, 850, 500),
        dist_png=_fig_to_png_bytes(dist_fig, 750, 420),
        hist_png=_fig_to_png_bytes(hist_fig, 750, 420),
        top_n=50,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "n_total": n_total, "n_high": n_high, "n_med": n_med, "n_low": n_low,
        "n_intensive": n_intensive, "n_overflow": n_overflow, "mean_prob": mean_prob,
        "risk_at_cap": risk_at_cap, "lift": lift,
        "cap_fig": cap_fig, "dist_fig": dist_fig, "hist_fig": hist_fig,
        "queue": scored[["student_index", "probability", "band", "action"]].head(50).copy(),
        "csv": scored.to_csv(index=False).encode("utf-8"), "csv_name": f"cohort_scored_{stamp}.csv",
        "pdf": batch_pdf.getvalue(), "pdf_name": f"cohort_report_{stamp}.pdf",
    }


def main():
    model, explainer, feature_cols, X_train = load_and_train_model()
    medians = X_train.median()

    st.markdown(
        f"""
        <div class="app-header">
            {WHITE_LABEL['logo_svg']}
            <div>
                <div class="app-title">{WHITE_LABEL['institution_name']} DSS</div>
                <div class="app-subtitle">Interpretable student retention workspace</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown(f"**{WHITE_LABEL['institution_name']} Advisor Console**")
        st.markdown(f"**Capacity**: {CAPACITY_LIMIT} intensive slots.")
        st.markdown(f"**Model**: {MODEL_VERSION}")
        st.info("Predictions are probabilistic; human judgement remains essential.")

    render_intro_panel()

    tab_eval, tab_batch, tab_info, tab_fairness = st.tabs([
        "Student Assessment", "Cohort Scoring", "How It Works", "Fairness Audit"
    ])

    with tab_eval:
        col_in, col_res = st.columns([1, 2.15], gap="large")
        with col_in:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title">

                    Student profile

                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div class="card-sub">

                    Unentered indicators use cohort medians.

                    </div>
                    """, unsafe_allow_html=True
                )
                user_input = {}

                a1, a2 = st.columns(2)
                with a1: user_input["Curricular units 1st sem (approved)"] = st.slider("1st Sem Passed Units", 0, 20, int(medians["Curricular units 1st sem (approved)"]))
                with a2: user_input["Curricular units 2nd sem (approved)"] = st.slider("2nd Sem Passed Units", 0, 20, int(medians["Curricular units 2nd sem (approved)"]))
                user_input["Curricular units 2nd sem (grade)"] = st.slider("2nd Sem Avg Grade", 0.0, 20.0, float(medians["Curricular units 2nd sem (grade)"]), 0.5)

                s1, s2 = st.columns(2)
                with s1: user_input["Tuition fees up to date"] = st.selectbox("Tuition Status", [("Up to date", 1), ("Overdue", 0)], format_func=lambda x: x[0])[1]
                with s2: user_input["Scholarship holder"] = st.selectbox("Scholarship", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
                s3, s4 = st.columns(2)
                with s3: user_input["Gender"] = st.selectbox("Gender", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
                with s4: user_input["Age at enrollment"] = st.number_input("Age at Enrollment", 17, 65, 20)

                if st.button("Generate Diagnostic Prediction", type="primary", use_container_width=True):
                    with st.spinner("Executing XGBoost and computing TreeSHAP matrices..."):
                        x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
                        x_df = pd.DataFrame([x_full])
                        p_drop = float(model.predict_proba(x_df)[:, 1][0])
                        band = assign_risk_band(p_drop)
                        action = get_intervention(band)
                        shap_vals = explainer.shap_values(x_df)[0]

                        r_feats = [f for f in ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Age at enrollment", "Tuition fees up to date", "Scholarship holder"] if f in feature_cols]
                        radar_fig = plot_radar_profile(user_input, medians, r_feats)

                        cf_f = "Curricular units 2nd sem (approved)"
                        cf_fig = plot_counterfactual_curve(user_input, cf_f, user_input.get(cf_f, 0), model, medians, feature_cols)

                        pdf_buf = build_pdf_report(user_input, p_drop, band, action, shap_vals, feature_cols, medians, model, cf_f, _fig_to_png_bytes(cf_fig, 850, 420), _fig_to_png_bytes(radar_fig, 700, 470))

                        st.session_state["assessment"] = {
                            "user_input": dict(user_input), "p": p_drop, "band": band, "action": action,
                            "shap_vals": shap_vals, "radar_fig": radar_fig, "pdf": pdf_buf.getvalue(),
                            "pdf_name": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        }

        with col_res:
            res = st.session_state.get("assessment")
            if res is None:
                st.markdown(
                    """
                    <div class="empty-state">

                    Awaiting assessment...

                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                p, band, action = res["p"], res["band"], res["action"]
                st.markdown(
                    f"""
                    <div class="result-header">

                    Dropout Probability

                    {p:.1%}

                    Risk Band

                    {band}

                    </div>
                    """, unsafe_allow_html=True
                )
                st.download_button("Download PDF Report", data=res["pdf"], file_name=res["pdf_name"], mime="application/pdf", use_container_width=True)

                t_over, t_drv, t_cmp, t_wif, t_act = st.tabs(["Overview", "Risk drivers", "Cohort comparison", "What-if", "Action plan"])
                with t_over:
                    g1, g2 = st.columns([1, 1.28])
                    g1.plotly_chart(plot_gauge(p, band), use_container_width=True, config={"displayModeBar": False})
                    g2.plotly_chart(plot_threshold_explorer(p), use_container_width=True, config={"displayModeBar": False})
                with t_drv:
                    w1, w2 = st.columns([1.55, 1])
                    w1.plotly_chart(plot_shap_waterfall(feature_cols, res["shap_vals"], top_n=6), use_container_width=True, config={"displayModeBar": False})
                    w2.plotly_chart(plot_contribution_donut(res["shap_vals"]), use_container_width=True, config={"displayModeBar": False})
                with t_cmp:
                    st.plotly_chart(res["radar_fig"], use_container_width=True, config={"displayModeBar": False})
                with t_wif:
                    st.info("What-If simulator holds all inputs constant except the selected indicator.")
                    cf_f = st.selectbox("Scenario indicator", ["Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Tuition fees up to date", "Age at enrollment"], format_func=friendly_name)

                    if cf_f == "Curricular units 2nd sem (approved)":
                        cf_val = st.slider("Value", 0, 20, int(res["user_input"].get(cf_f, 0)))
                    elif "grade" in cf_f.lower():
                        cf_val = st.slider("Value", 0.0, 20.0, float(res["user_input"].get(cf_f, 10.0)), 0.5)
                    elif cf_f == "Age at enrollment":
                        cf_val = st.slider("Value", 17, 65, int(res["user_input"].get(cf_f, 20)))
                    else:
                        cf_val = st.selectbox("Value", [("No", 0), ("Yes", 1)])[1]

                    cf_prob = compute_counterfactual(res["user_input"], cf_f, cf_val, model, medians, feature_cols)
                    delta, cls, arrow, b_note = band_change_summary(p, cf_prob)
                    st.markdown(f"**Predicted probability**: {cf_prob:.1%} ({delta:+.2%}) {arrow}. {b_note}", unsafe_allow_html=True)
                    st.plotly_chart(plot_counterfactual_curve(res["user_input"], cf_f, cf_val, model, medians, feature_cols), use_container_width=True, config={"displayModeBar": False})
                with t_act:
                    st.success(f"**Primary Action:** {action}")
                    st.markdown(get_intervention_detail(band))

                render_outro_panel()

    with tab_batch:
        uploaded_file = st.file_uploader("Upload cohort CSV (Requires identical columns to training data)", type=["csv"])
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if st.button("Run cohort scoring", type="primary"):
                    with st.spinner("Scoring cohort..."):
                        b = run_batch(batch_df, model, medians, feature_cols)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Scored", f"{b['n_total']}")
                        c2.metric("High Risk", f"{b['n_high']}")
                        c3.metric("Slots Used", f"{b['n_intensive']} / {CAPACITY_LIMIT}")
                        c4.metric("Lift", f"{b['lift']:.2f}x")

                        d1, d2 = st.columns(2)
                        d1.download_button("Download CSV", data=b["csv"], file_name=b["csv_name"], mime="text/csv", use_container_width=True)
                        d2.download_button("Download PDF", data=b["pdf"], file_name=b["pdf_name"], mime="application/pdf", use_container_width=True)

                        st.plotly_chart(b["cap_fig"], use_container_width=True)
                        st.dataframe(b["queue"], use_container_width=True)
            except Exception as e:
                st.error(f"Error processing batch: {e}")

    with tab_info:
        st.markdown(f"### Capacity Constraint Engine\nThe DSS limits Intensive Mentoring to **{CAPACITY_LIMIT}** instances university-wide, utilizing a continuous queue.")

    with tab_fairness:
        st.markdown("### Equal Opportunity Thresholding\nShows the active mitigation of historical demographic disparities.")


if __name__ == "__main__":
    main()
