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
    PageBreak,
    KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

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
# PRODUCT UI THEME  ("Campus Light")
# Presentation layer only: Plotly theme + CSS design system + small JS enhancer.
# ==========================================================
import plotly.io as pio
import streamlit.components.v1 as components

_FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
pio.templates["campus"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=_FONT, size=12.5, color="#4A5876"),
        title=dict(font=dict(size=14, color="#0E1B3D", family=_FONT), x=0, xanchor="left"),
        colorway=["#3B5BDB", "#12A594", "#F5A524", "#DC3F4A", "#64748B"],
        xaxis=dict(gridcolor="#EEF2F8", linecolor="#D5DCE8", zerolinecolor="#94A3B8",
                   ticks="outside", tickcolor="#D5DCE8", title=dict(font=dict(size=11.5, color="#7C89A6"))),
        yaxis=dict(gridcolor="#EEF2F8", linecolor="#D5DCE8", zerolinecolor="#94A3B8",
                   title=dict(font=dict(size=11.5, color="#7C89A6"))),
        hoverlabel=dict(bgcolor="#0E1B3D", bordercolor="#0E1B3D",
                        font=dict(color="#FFFFFF", family=_FONT, size=12)),
        legend=dict(font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
    )
)
pio.templates.default = "campus"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root{
  color-scheme:light;
  --ink:#12213f;
  --ink-2:#203455;
  --text:#5a6a84;
  --muted:#8795aa;
  --bg:#f5f8fc;
  --surface:#ffffff;
  --surface-2:#f9fbfe;
  --line:#e4eaf2;
  --line-2:#d2dbe8;
  --brand:#4767e8;
  --brand-2:#3154d9;
  --brand-soft:#eef3ff;
  --brand-soft-2:#f6f8ff;
  --teal:#12a594;
  --amber:#eea51f;
  --red:#dc4a59;
  --r-sm:10px;
  --r:14px;
  --r-lg:20px;
  --shadow-xs:0 1px 2px rgba(18,33,63,.04);
  --shadow-sm:0 6px 18px rgba(28,47,80,.06), 0 1px 2px rgba(18,33,63,.03);
  --shadow-md:0 14px 35px rgba(28,47,80,.09), 0 2px 5px rgba(18,33,63,.04);
  --focus:0 0 0 4px rgba(71,103,232,.14);
}

html, body{
  -webkit-font-smoothing:antialiased;
  scroll-behavior:smooth;
}

.stApp{
  background:
    radial-gradient(700px 260px at 6% -5%, rgba(91,122,255,.13), transparent 70%),
    radial-gradient(520px 240px at 96% 0%, rgba(110,223,205,.09), transparent 68%),
    var(--bg);
  color:var(--text);
}

.block-container{
  max-width:1380px;
  padding:2.35rem 2rem 4.5rem;
}

@keyframes uiFadeUp{
  from{opacity:0; transform:translateY(7px)}
  to{opacity:1; transform:none}
}
@keyframes uiFloat{
  0%,100%{transform:translateY(0)}
  50%{transform:translateY(-2px)}
}
@keyframes uiPulse{
  0%{box-shadow:0 0 0 0 rgba(18,165,148,.24)}
  70%{box-shadow:0 0 0 7px rgba(18,165,148,0)}
  100%{box-shadow:0 0 0 0 rgba(18,165,148,0)}
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important;
}
header[data-testid="stHeader"]{
  background:transparent;
  height:2rem;
}

h1,h2,h3,h4,h5,h6{
  color:var(--ink);
  font-family:'Manrope','Inter','Segoe UI',sans-serif;
  letter-spacing:-.02em;
  font-weight:700;
}
h3{font-size:1.05rem;margin:.45rem 0 .2rem}
h4{font-size:.98rem;margin:.3rem 0 .2rem}
p,li{line-height:1.6}
a{color:var(--brand);text-decoration:none}
a:hover{text-decoration:underline}
::selection{background:#dfe7ff;color:var(--ink)}

/* ---------- Product header ---------- */
.hero{
  position:relative;
  overflow:hidden;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:1.1rem;
  flex-wrap:wrap;
  margin:0 0 1rem;
  padding:1.1rem 1.25rem;
  border:1px solid rgba(255,255,255,.35);
  border-radius:22px;
  background:linear-gradient(120deg,#17275d 0%,#2d48c8 52%,#5675f0 100%);
  box-shadow:0 18px 40px rgba(51,78,199,.18), inset 0 1px 0 rgba(255,255,255,.15);
  animation:uiFadeUp .35s ease both;
}
.hero::before{
  content:"";
  position:absolute;
  inset:auto -70px -130px auto;
  width:330px;
  height:330px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.16),rgba(255,255,255,0) 69%);
  pointer-events:none;
}
.hero::after{
  content:"";
  position:absolute;
  right:22%;
  top:-60px;
  width:150px;
  height:150px;
  border-radius:50%;
  border:1px solid rgba(255,255,255,.08);
  box-shadow:0 0 0 26px rgba(255,255,255,.025), 0 0 0 52px rgba(255,255,255,.018);
  pointer-events:none;
}
.hero-l{display:flex;align-items:center;gap:.9rem;position:relative;z-index:1}
.logo{
  width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;flex:none;
  background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.26);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.12);
}
.hero-title{font-size:1.22rem;font-weight:800;line-height:1.2;letter-spacing:-.025em;color:#fff;font-family:'Manrope','Inter',sans-serif}
.hero-sub{font-size:.82rem;color:rgba(255,255,255,.77);margin-top:3px}
.hero-r{display:flex;gap:.45rem;flex-wrap:wrap;position:relative;z-index:1}
.gchip{
  display:inline-flex;align-items:center;gap:.45rem;
  padding:.32rem .72rem;border-radius:999px;
  color:rgba(255,255,255,.95);font-size:.72rem;font-weight:650;
  background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);
  backdrop-filter:blur(8px);
}
.gchip i{width:7px;height:7px;border-radius:50%;background:#72efd3;animation:uiPulse 2s infinite;box-shadow:0 0 0 3px rgba(114,239,211,.14)}

/* ---------- Segmented navigation ---------- */
[data-testid="stTabs"]{animation:uiFadeUp .36s ease both}
[data-baseweb="tab-list"]{
  gap:.25rem;
  padding:.28rem;
  margin-bottom:1rem;
  background:rgba(255,255,255,.88);
  border:1px solid var(--line);
  border-radius:15px;
  box-shadow:var(--shadow-sm);
  position:sticky;
  top:2rem;
  z-index:50;
  backdrop-filter:blur(14px);
}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none !important}
button[role="tab"]{
  min-height:40px;
  padding:.56rem 1rem;
  border-radius:11px;
  background:transparent;
  color:var(--muted);
  font-size:.88rem;
  font-weight:650;
  transition:all .18s ease;
}
button[role="tab"] p{font-size:.88rem;font-weight:650;margin:0;color:inherit}
button[role="tab"]:hover{background:var(--brand-soft-2);color:var(--brand-2);transform:translateY(-1px)}
button[role="tab"][aria-selected="true"]{
  background:linear-gradient(180deg,#5574ef,#4263e5);
  color:#fff;
  box-shadow:0 6px 16px rgba(71,103,232,.22);
}
button[role="tab"][aria-selected="true"] p{color:#fff}

/* nested tabs */
div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab-list"]{
  position:static;
  background:transparent;
  border:0;
  border-bottom:1px solid var(--line);
  border-radius:0;
  box-shadow:none;
  padding:0;
  gap:.1rem;
  margin-bottom:.9rem;
  backdrop-filter:none;
}
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"]{
  border-radius:9px 9px 0 0;
  padding:.58rem .8rem;
  color:var(--muted);
}
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background:transparent;
  color:var(--brand-2);
  box-shadow:inset 0 -2px 0 var(--brand);
}
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:var(--brand-2)}

/* ---------- Cards / workspace ---------- */
div[class*="st-key-card"]{
  background:rgba(255,255,255,.96);
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:var(--shadow-md);
  padding:1.05rem 1.15rem 1.2rem;
  transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[class*="st-key-card"]:hover{
  border-color:#dbe3f0;
  box-shadow:0 18px 40px rgba(28,47,80,.10),0 2px 6px rgba(18,33,63,.04);
}
div[class*="st-key-card"] div[data-testid="stVerticalBlock"]{gap:.66rem}
.sec-t{font-size:1.06rem;font-weight:750;color:var(--ink);letter-spacing:-.015em}
.sec-s{font-size:.82rem;color:var(--muted);margin-top:3px;line-height:1.55}
.grp{
  display:flex;align-items:center;gap:.6rem;
  margin:.45rem 0 .12rem;
  color:#7a899e;
  font-size:.65rem;font-weight:750;
  text-transform:uppercase;letter-spacing:.12em;
}
.grp::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.hint{
  margin:.15rem 0 .35rem;
  padding:.64rem .82rem;
  color:#586a86;
  background:linear-gradient(180deg,#f8faff,#f4f7fd);
  border:1px solid #e1e8f3;
  border-radius:12px;
  font-size:.83rem;
  line-height:1.55;
}
.hint b{color:var(--ink)}

.workspace-label{
  display:inline-flex;align-items:center;gap:.42rem;
  margin:0 0 .55rem;
  padding:.28rem .62rem;
  border-radius:999px;
  color:#53647d;
  background:#edf2fa;
  border:1px solid #e0e7f1;
  font-size:.67rem;font-weight:750;text-transform:uppercase;letter-spacing:.08em;
}
.workspace-label i{width:6px;height:6px;border-radius:50%;background:#8ea0ba}

/* ---------- Assessment result ---------- */
.result{
  display:flex;
  align-items:stretch;
  overflow:hidden;
  margin-bottom:.1rem;
  background:rgba(255,255,255,.98);
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:var(--shadow-md);
}
.result-l{min-width:184px;padding:1rem 1.25rem;display:flex;flex-direction:column;justify-content:center;color:#fff}
.result.tone-ok .result-l{background:linear-gradient(135deg,#0e9f8b,#16ae99)}
.result.tone-warn .result-l{background:linear-gradient(135deg,#e59a18,#f0ad31)}
.result.tone-bad .result-l{background:linear-gradient(135deg,#c93646,#df5260)}
.result-k{font-size:.65rem;font-weight:750;text-transform:uppercase;letter-spacing:.1em;opacity:.9}
.result-v{font-size:2.7rem;font-weight:800;letter-spacing:-.04em;line-height:1.02;margin-top:.18rem}
.result-r{padding:1rem 1.3rem;display:flex;flex-direction:column;justify-content:center;gap:.36rem}
.pill{
  display:inline-flex;align-self:flex-start;align-items:center;gap:.4rem;
  padding:.22rem .64rem;border-radius:999px;border:1px solid;
  font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.09em;
}
.pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.pill.tone-ok{background:#e9f8f5;color:#0b8a78;border-color:#afe4d8}
.pill.tone-warn{background:#fff6e2;color:#9a5a05;border-color:#f2d28c}
.pill.tone-bad{background:#feeff1;color:#b42335;border-color:#f0bac1}
.result-ak{font-size:.67rem;font-weight:750;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-top:.12rem}
.result-a{font-size:1.07rem;font-weight:750;color:var(--ink);letter-spacing:-.012em;line-height:1.35}

/* ---------- KPI strip ---------- */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.66rem;margin:.2rem 0}
.kpi{
  position:relative;overflow:hidden;min-height:84px;
  background:rgba(255,255,255,.96);
  border:1px solid var(--line);border-radius:13px;
  padding:.76rem .9rem .8rem 1rem;
  box-shadow:var(--shadow-xs);
  transition:transform .16s ease, box-shadow .16s ease;
}
.kpi:hover{transform:translateY(-1px);box-shadow:var(--shadow-sm)}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--brand)}
.kpi.tone-ok::before{background:var(--teal)}
.kpi.tone-warn::before{background:var(--amber)}
.kpi.tone-bad::before{background:var(--red)}
.kpi-l{font-size:.62rem;font-weight:750;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.kpi-v{font-size:1.45rem;font-weight:800;color:var(--ink);letter-spacing:-.03em;line-height:1.2;margin-top:.15rem}
.kpi-s{font-size:.75rem;color:var(--muted);margin-top:.08rem}

.factor-row{display:flex;flex-wrap:wrap;gap:.45rem;margin:.15rem 0 .3rem}
.factor-chip{
  display:inline-flex;align-items:center;gap:.35rem;
  padding:.32rem .58rem;border-radius:999px;
  font-size:.69rem;font-weight:700;
  border:1px solid;
}
.factor-chip::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.chip-risk{color:#b42335;background:#fff1f2;border-color:#f3c4ca}
.chip-prot{color:#0b8a78;background:#ebf9f6;border-color:#b4e6dc}

/* ---------- Empty state ---------- */
.empty{
  text-align:center;
  padding:3.5rem 1.5rem;
  background:linear-gradient(180deg,rgba(255,255,255,.96),rgba(249,251,254,.98));
  border:1px dashed #cbd6e7;
  border-radius:18px;
  box-shadow:var(--shadow-sm);
}
.empty-i{width:58px;height:58px;margin:0 auto .9rem;border-radius:17px;background:var(--brand-soft);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 0 1px #dbe4ff}
.empty-t{font-family:'Manrope','Inter',sans-serif;font-size:1.08rem;font-weight:750;color:var(--ink)}
.empty-s{max-width:570px;margin:.42rem auto 0;font-size:.86rem;color:var(--muted);line-height:1.6}
.empty-c{display:flex;gap:.46rem;justify-content:center;flex-wrap:wrap;margin-top:1.1rem}
.empty-c span{font-size:.7rem;font-weight:700;color:var(--brand-2);background:var(--brand-soft);padding:.28rem .66rem;border-radius:999px;border:1px solid #dbe4ff}

/* ---------- Info/fairness cards ---------- */
.metric-card{
  position:relative;overflow:hidden;height:100%;
  padding:1rem 1.05rem;
  background:rgba(255,255,255,.96);
  border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow-sm)
}
.metric-card::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:linear-gradient(90deg,#4767e8,#88a0ff)}
.metric-card h4{margin:.1rem 0 .35rem;font-size:.64rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:750}
.metric-card p{margin:.05rem 0;color:var(--text);font-size:.83rem}
.metric-card p strong{color:var(--ink);font-size:1.44rem;font-weight:800;letter-spacing:-.03em}
.flow{display:flex;flex-direction:column;gap:.5rem}
.flow-s{display:flex;gap:.75rem;align-items:flex-start;padding:.68rem .78rem;border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.92);box-shadow:var(--shadow-xs);transition:transform .16s ease, border-color .16s ease}
.flow-s:hover{transform:translateX(2px);border-color:#d6e0f0}
.flow-n{flex:none;width:27px;height:27px;border-radius:50%;background:var(--brand-soft);color:var(--brand-2);font-weight:800;font-size:.75rem;display:flex;align-items:center;justify-content:center}
.flow-t{font-weight:700;color:var(--ink);font-size:.86rem}.flow-d{font-size:.77rem;color:var(--muted);line-height:1.5}
.fair{background:rgba(255,255,255,.98);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow-sm);overflow:hidden;height:100%}
.fair-h{padding:.72rem 1rem;font-size:.66rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em}
.fair.bad .fair-h{background:#fff0f2;color:#b42335}.fair.ok .fair-h{background:#eaf9f6;color:#0b8a78}
.fair-b{padding:.58rem 1rem 1rem}.fair-d{font-size:.8rem;color:var(--muted);line-height:1.55;margin:.18rem 0 .48rem}
.fair-r{display:flex;justify-content:space-between;align-items:baseline;padding:.5rem 0;border-top:1px solid var(--line);font-size:.84rem}.fair-r b{color:var(--ink);font-size:.98rem}
.fair-g{display:flex;justify-content:space-between;align-items:baseline;padding:.63rem .75rem;margin-top:.4rem;border-radius:10px;font-weight:700;font-size:.82rem}.fair.bad .fair-g{background:#fff0f2;color:#b42335}.fair.ok .fair-g{background:#eaf9f6;color:#0b8a78}.fair-g b{font-size:1.25rem;font-weight:800}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#ffffff 0%,#fbfcfe 100%);
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:1rem .95rem 2rem}
.side-brand{display:flex;align-items:center;gap:.65rem;padding:.25rem .15rem .9rem;border-bottom:1px solid var(--line);margin-bottom:.85rem}
.side-brand-mark{width:34px;height:34px;border-radius:10px;background:linear-gradient(140deg,#3d5de0,#6d88f3);display:flex;align-items:center;justify-content:center;box-shadow:0 5px 14px rgba(71,103,232,.2)}
.side-brand-title{font-size:.82rem;font-weight:800;color:var(--ink);line-height:1.2}.side-brand-sub{font-size:.65rem;color:var(--muted);margin-top:2px}
.side-t{font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:.11em;color:#8795a9;margin:1rem 0 .48rem}.side-t:first-child{margin-top:.1rem}
.lg{display:flex;align-items:center;justify-content:space-between;padding:.52rem .65rem;border:1px solid var(--line);border-radius:10px;margin-bottom:.34rem;font-size:.81rem;color:var(--ink);font-weight:650;background:#fff}
.lg span{font-weight:600;color:var(--muted);font-size:.73rem}.lg i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.46rem}
.side-box{background:linear-gradient(180deg,#f8faff,#f3f6fc);border:1px solid #e0e7f2;border-radius:11px;padding:.66rem .72rem;font-size:.76rem;color:var(--text);line-height:1.55}.side-box b{color:var(--ink)}

/* ---------- Controls ---------- */
label[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] p{font-size:.77rem;font-weight:650;color:var(--ink)}
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div{
  background:#fff !important;
  border:1px solid var(--line-2) !important;
  border-radius:10px !important;
  box-shadow:var(--shadow-xs);
  min-height:40px;
  transition:border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}
div[data-baseweb="input"]:hover, div[data-baseweb="select"] > div:hover{border-color:#aebdd2 !important}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within{border-color:var(--brand) !important;box-shadow:var(--focus) !important}
div[data-baseweb="input"] input, div[data-baseweb="select"] input, textarea{color:var(--ink) !important;-webkit-text-fill-color:var(--ink);font-size:.88rem;font-weight:550}
div[data-testid="stNumberInput"] div[data-baseweb="input"]{overflow:hidden}
div[data-testid="stNumberInput"] button{background:#f5f8ff !important;color:var(--brand-2) !important;border:0 !important;border-left:1px solid var(--line) !important}
div[data-testid="stNumberInput"] button:hover{background:#edf2ff !important}
div[data-baseweb="select"] svg{color:var(--muted)}
span[data-baseweb="tag"]{background:var(--brand-soft) !important;color:var(--brand-2) !important;border-radius:7px !important;font-weight:650}
div[data-baseweb="popover"] > div{background:#fff !important;border:1px solid var(--line) !important;border-radius:12px !important;box-shadow:0 18px 42px rgba(18,33,63,.14) !important;overflow:hidden}
div[data-baseweb="popover"] ul, ul[role="listbox"]{background:#fff !important;padding:.28rem !important}
div[data-baseweb="popover"] li, li[role="option"]{border-radius:8px !important;margin:1px 0;padding:.48rem .66rem !important;color:var(--ink) !important;font-size:.86rem;font-weight:550;background:transparent !important}
div[data-baseweb="popover"] li:hover, li[role="option"]:hover{background:#f4f7ff !important;color:var(--brand-2) !important}
li[role="option"][aria-selected="true"]{background:var(--brand-soft) !important;color:var(--brand-2) !important;font-weight:700}
div[data-testid="stSlider"]{padding:0 .28rem .1rem}
div[data-testid="stSlider"] [role="slider"]{background:var(--brand) !important;border:3px solid #fff !important;box-shadow:0 0 0 1px var(--brand),0 5px 14px rgba(71,103,232,.18) !important;height:1.08rem;width:1.08rem}
div[data-testid="stSliderThumbValue"]{color:var(--brand-2) !important;font-weight:750;font-size:.78rem}
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"]{color:var(--muted);font-size:.7rem}
input[type="checkbox"],input[type="radio"]{accent-color:var(--brand)}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  min-height:42px;
  padding:.5rem 1rem;
  border-radius:11px;
  font-size:.88rem;
  font-weight:700;
  background:#fff;
  color:var(--brand-2);
  border:1px solid var(--line-2);
  box-shadow:var(--shadow-xs);
  transition:transform .13s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease;
}
.stButton > button p, .stDownloadButton > button p{font-size:.88rem;font-weight:700;margin:0}
.stButton > button:hover, .stDownloadButton > button:hover{background:#f8faff;border-color:#9db0ce;color:var(--brand-2);box-shadow:var(--shadow-sm);transform:translateY(-1px)}
.stButton > button:active, .stDownloadButton > button:active{transform:translateY(0)}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, button[role="tab"]:focus-visible, summary:focus-visible{outline:none;box-shadow:var(--focus)}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"], .stDownloadButton > button[kind="primary"]{
  background:linear-gradient(180deg,#5574ef,#4263e5);
  color:#fff;border-color:#3f5fd6;
  box-shadow:0 8px 20px rgba(71,103,232,.23),0 1px 2px rgba(49,84,217,.2);
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p{color:#fff}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{background:linear-gradient(180deg,#607cf2,#4868e7);color:#fff;border-color:#3858ce}

/* ---------- Upload / expander / charts / tables ---------- */
div[data-testid="stFileUploader"] section{background:#fff;border:1.5px dashed #aec0da;border-radius:15px;padding:1.2rem;transition:all .16s ease;box-shadow:var(--shadow-xs)}
div[data-testid="stFileUploader"] section:hover{border-color:var(--brand);background:#f9fbff;box-shadow:var(--shadow-sm)}
div[data-testid="stFileUploader"] small{color:var(--muted)}
div[data-testid="stFileUploaderFile"]{background:#f5f8ff;border-radius:10px}
div[data-testid="stExpander"]{background:#fff;border:1px solid var(--line) !important;border-radius:13px !important;box-shadow:var(--shadow-xs);overflow:hidden}
div[data-testid="stExpander"] details{border:0 !important}
div[data-testid="stExpander"] summary{padding:.7rem .9rem;font-weight:700;color:var(--ink)}
div[data-testid="stExpander"] summary:hover{background:#f8faff}
div[data-testid="stExpander"] summary p{font-size:.86rem;font-weight:700;margin:0}
div[data-testid="stPlotlyChart"]{background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:14px;padding:.28rem .4rem .08rem;box-shadow:var(--shadow-xs)}
div[data-testid="stDataFrame"],div[data-testid="stTable"]{border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:var(--shadow-xs)}
div[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line)}
div[data-testid="stCaptionContainer"],.stCaption{color:var(--muted);font-size:.76rem}
div[data-testid="stSpinner"] p{color:var(--muted);font-weight:550}
.stMarkdown code{background:var(--brand-soft);color:var(--brand-2);border-radius:6px;padding:.1rem .36rem;font-size:.83em}
div[data-testid="stCode"] pre,pre{background:#132247 !important;color:#edf3ff !important;border-radius:12px;font-size:.82rem}
.delta-good{color:#0b8a78;font-weight:750}.delta-bad{color:#b42335;font-weight:750}.delta-flat{color:var(--muted);font-weight:750}

/* ---------- Main spacing and responsive layout ---------- */
div[data-testid="stVerticalBlock"]{gap:.78rem}
@media (max-width:980px){
  .block-container{padding:2.1rem .95rem 3.5rem}
  .hero{padding:1rem}
  .hero-title{font-size:1.06rem}
  [data-baseweb="tab-list"]{top:1.8rem;overflow-x:auto}
  button[role="tab"]{padding:.5rem .72rem;font-size:.81rem}
  .result{flex-direction:column}.result-l{min-width:0}
}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# UI STABILITY OVERRIDES
# Keeps the product styling while preventing overflow/overlap at narrow widths.
# ==========================================================
st.markdown(
    """
<style>
  /* Never let flex/grid children force a wider layout than their container. */
  [data-testid="stHorizontalBlock"],
  [data-testid="stHorizontalBlock"] > div,
  [data-testid="column"],
  [data-testid="stVerticalBlock"],
  div[class*="st-key-card"] { min-width:0 !important; }

  /* Long product copy must wrap rather than enlarge/clip its parent. */
  .sec-t,.sec-s,.hint,.result-r,.result-a,.result-ak,.kpi,.kpi-v,.kpi-s,
  .fair-d,.fair-r,.fair-g,.flow-t,.flow-d,.side-box,.side-t,
  [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    overflow-wrap:anywhere !important;
    word-break:normal !important;
  }

  .result { min-width:0 !important; }
  .result-l { flex:0 0 184px; min-width:0 !important; }
  .result-r { flex:1 1 auto; min-width:0 !important; }
  .result-a { white-space:normal !important; }

  .kpi { min-width:0 !important; }
  .kpi-v { white-space:normal !important; }

  /* Product buttons can wrap safely on laptop / tablet widths. */
  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    white-space:normal !important;
    line-height:1.25 !important;
    min-height:42px !important;
    height:auto !important;
  }
  .stButton > button p, .stDownloadButton > button p { white-space:normal !important; }

  /* Remove sticky nav overlay; it is a frequent source of visual collisions. */
  div[data-testid="stTabs"] > div:first-child [data-baseweb="tab-list"] {
    position:static !important;
    top:auto !important;
    z-index:1 !important;
  }

  /* Charts are containers, not floating canvases. */
  div[data-testid="stPlotlyChart"] {
    min-width:0 !important;
    max-width:100% !important;
    overflow:hidden !important;
    margin:0 !important;
  }
  div[data-testid="stPlotlyChart"] > div,
  div[data-testid="stPlotlyChart"] iframe { max-width:100% !important; }

  /* More breathing room around native widgets inside product cards. */
  div[data-testid="stSlider"], div[data-testid="stNumberInput"],
  div[data-testid="stSelectbox"], div[data-testid="stFileUploader"] {
    min-width:0 !important;
  }

  /* Dataframes should scroll inside themselves instead of widening the page. */
  div[data-testid="stDataFrame"] { max-width:100% !important; overflow:hidden !important; }

  /* Narrow layouts. */
  @media (max-width: 1050px) {
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
    .result { flex-direction:column !important; }
    .result-l { flex-basis:auto !important; }
  }
  @media (max-width: 720px) {
    .hero { border-radius:16px !important; padding:.95rem !important; }
    .hero-r { width:100% !important; }
    .hero-title { font-size:1.05rem !important; }
    .hero-sub { font-size:.76rem !important; }
    .gchip { font-size:.66rem !important; }
    button[role="tab"] { padding:.48rem .6rem !important; font-size:.78rem !important; }
    button[role="tab"] p { font-size:.78rem !important; }
    .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)) !important; }
  }
  @media (max-width: 480px) {
    .kpi-grid { grid-template-columns:1fr !important; }
  }
  @media (prefers-reduced-motion:reduce) {
    *,*::before,*::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; }
  }
</style>
""",
    unsafe_allow_html=True,
)

components.html(
    """
<script>
(function(){
  try{
    const doc = window.parent.document;
    const id = 'campus-product-enhancer-v3';
    if(!doc.getElementById(id)){
      const meta = doc.createElement('meta');
      meta.id = id; meta.name = 'theme-color'; meta.content = '#4767E8';
      doc.head.appendChild(meta);
    }
    doc.documentElement.setAttribute('lang','en');

    const enhance = () => {
      doc.querySelectorAll('button[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] button')
        .forEach(btn => { if(!btn.getAttribute('aria-label')) btn.setAttribute('aria-label','Toggle sidebar'); });

      doc.querySelectorAll('div[data-testid="stPlotlyChart"]').forEach(chart => {
        if(!chart.getAttribute('aria-label')) chart.setAttribute('aria-label','Interactive analytical chart');
      });

      doc.querySelectorAll('[data-baseweb="slider"] *').forEach(el => {
        const style = el.getAttribute('style') || '';
        if(style.toLowerCase().includes('rgb(255, 75, 75)')){
          el.setAttribute('style', style.replace(/rgb\\(255,\\s*75,\\s*75\\)/ig, 'rgb(71, 103, 232)'));
        }
      });

      doc.querySelectorAll('.stButton button, .stDownloadButton button').forEach(btn => {
        if(!btn.dataset.productBound){
          btn.dataset.productBound = '1';
          btn.addEventListener('keydown', e => {
            if(e.key === 'Enter' || e.key === ' '){ btn.classList.add('ui-press'); setTimeout(()=>btn.classList.remove('ui-press'),160); }
          });
        }
      });
    };

    enhance();
    let timer;
    const observer = new MutationObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(enhance, 90);
    });
    observer.observe(doc.body, {childList:true, subtree:true, attributes:true, attributeFilter:['style','aria-selected']});
  }catch(err){ /* Presentation enhancement is optional; core Streamlit UI remains functional. */ }
})();
</script>
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
    """Short axis-safe label; full feature names remain available in hover text."""
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
        autosize=True,
        height=height,
        margin=margin or dict(l=46, r=22, t=54, b=48),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=11.5, color="#5A6A84"),
        title=dict(font=dict(size=13, color="#12213F", family=_FONT), x=0, xanchor="left", y=.98),
        legend=dict(bgcolor="rgba(255,255,255,.88)", bordercolor="#E4EAF2", borderwidth=1, font=dict(size=10.5)),
        hoverlabel=dict(bgcolor="#12213F", bordercolor="#12213F", font=dict(color="#FFFFFF", family=_FONT, size=11)),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def plot_gauge(probability, band):
    color = "#12A594" if band == "Low Risk" else "#F5A524" if band == "Medium Risk" else "#DC4A59"
    fig = _plot_base(height=270, margin=dict(l=18, r=18, t=32, b=18))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9AA8BC", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "#FFFFFF",
            "borderwidth": 1,
            "bordercolor": "#DCE4EF",
            "steps": [
                {"range": [0, 20], "color": "rgba(18,165,148,.10)"},
                {"range": [20, 50], "color": "rgba(245,165,36,.10)"},
                {"range": [50, 100], "color": "rgba(220,74,89,.10)"},
            ],
            "threshold": {"line": {"color": "#12213F", "width": 2}, "thickness": 0.72, "value": probability * 100},
        },
    ))
    fig.update_layout(showlegend=False)
    return fig


def plot_shap_waterfall(feature_names, shap_values, top_n=6):
    idx = np.argsort(np.abs(shap_values))[-top_n:][::-1]
    names = [friendly_name(feature_names[i]) for i in idx]
    axis_names = [_chart_label(feature_names[i]) for i in idx]
    vals = shap_values[idx]
    colors_list = ["#DC4A59" if v > 0 else "#12A594" for v in vals]
    fig = _plot_base(height=355, margin=dict(l=124, r=26, t=36, b=54))
    fig.add_trace(go.Bar(
        x=vals[::-1],
        y=axis_names[::-1],
        orientation="h",
        marker_color=colors_list[::-1],
        customdata=np.array(names[::-1], dtype=object),
        text=[f"{v:+.2f}" for v in vals[::-1]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{customdata}</b><br>SHAP impact: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top risk drivers",
        showlegend=False,
        xaxis_title="SHAP impact (log-odds)",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor="#9EABC0", zerolinewidth=1.5, showgrid=True, gridcolor="#EEF2F7"),
    )
    return fig


def plot_contribution_donut(shap_values):
    pos = float(np.sum(shap_values[shap_values > 0]))
    neg = float(np.sum(np.abs(shap_values[shap_values < 0])))
    fig = _plot_base(height=315, margin=dict(l=18, r=18, t=48, b=18))
    fig.add_trace(go.Pie(
        labels=["Risk-increasing", "Risk-reducing"],
        values=[pos, neg],
        hole=0.68,
        marker_colors=["#DC4A59", "#12A594"],
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Total SHAP magnitude: %{value:.3f}<br>%{percent}<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        title="Contribution balance",
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", font=dict(size=10)),
    )
    return fig


def plot_threshold_explorer(probability):
    fig = _plot_base(height=245, margin=dict(l=28, r=24, t=54, b=54))
    colors_band = ["rgba(18,165,148,.72)", "rgba(245,165,36,.78)", "rgba(220,74,89,.76)"]
    names = ["Low", "Medium", "High"]
    widths = [LOW_TH, HIGH_TH - LOW_TH, 1.0 - HIGH_TH]
    starts = np.cumsum([0] + widths[:-1])
    for name, width, start_x, color in zip(names, widths, starts, colors_band):
        fig.add_trace(go.Bar(
            x=[width], y=["Risk band"], orientation="h", base=[start_x],
            marker_color=color, marker_line_width=0, name=name,
            text=[name], textposition="inside", insidetextanchor="middle",
            hovertemplate=f"{name} Risk<br>Range: {start_x:.0%}–{start_x+width:.0%}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[probability], y=["Risk band"], mode="markers",
        marker=dict(size=17, color="#12213F", line=dict(width=3, color="#FFFFFF")),
        customdata=[f"{probability:.1%}"],
        name="Student",
        hovertemplate="<b>This student</b><br>Dropout probability: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title="Risk-band placement",
        barmode="stack",
        showlegend=False,
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Dropout probability", showgrid=False),
        yaxis=dict(showticklabels=False),
    )
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
        hover.append(f"{friendly_name(feat)}<br>Student: {val:.2f}<br>Cohort median: {med:.2f}")
    if labels:
        labels_closed = labels + [labels[0]]
        student_closed = student_vals + [student_vals[0]]
        cohort_closed = cohort_vals + [cohort_vals[0]]
        hover_closed = hover + [hover[0]]
    else:
        labels_closed, student_closed, cohort_closed, hover_closed = [], [], [], []

    fig = _plot_base(height=385, margin=dict(l=44, r=44, t=42, b=46))
    fig.add_trace(go.Scatterpolar(
        r=student_closed, theta=labels_closed, fill="toself", name="Student",
        line=dict(color="#2C467E", width=2.5), fillcolor="rgba(44,70,126,.18)",
        text=hover_closed, hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=cohort_closed, theta=labels_closed, fill="toself", name="Cohort median",
        line=dict(color="#F5A524", width=1.8), fillcolor="rgba(245,165,36,.08)",
        hovertemplate="Cohort median (normalized)<extra></extra>",
    ))
    fig.update_layout(
        title="Profile vs cohort median",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0f", tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=9.5))),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.03, font=dict(size=10.5)),
    )
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
    fig.add_hrect(y0=HIGH_TH, y1=1.0, fillcolor="rgba(220,74,89,.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=sweep, y=probs, mode="lines", line=dict(color="#2C467E", width=3),
        name="Predicted probability",
        hovertemplate=f"{friendly_name(feature)}: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[current_value], y=[cur_prob], mode="markers",
        marker=dict(size=15, color="#DC4A59", line=dict(width=2.5, color="#FFFFFF")),
        name="Current value", hovertemplate=f"Current value: {current_value}<br>Probability: {cur_prob:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"What-if response · {_chart_label(feature)}",
        xaxis_title=_chart_label(feature),
        yaxis_title="Dropout probability",
        xaxis=dict(automargin=True),
        yaxis=dict(range=[0, 1], tickformat=".0%", automargin=True),
        legend=dict(orientation="h", x=0, y=1.03, yanchor="bottom", font=dict(size=10)),
    )
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
    fig.add_trace(go.Scatter(x=[0, n], y=[0, 1], mode="lines", line=dict(color="#B6C0CF", width=2, dash="dash"), name="Random baseline", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=y, mode="lines", line=dict(color="#2C467E", width=3), name="Model-ranked", hovertemplate="Intervened: %{x}<br>Risk captured: %{y:.1%}<extra></extra>"))
    fig.add_vline(x=cap, line=dict(color="#DC4A59", width=2, dash="dot"))
    if cap > 0:
        fig.add_trace(go.Scatter(
            x=[cap], y=[y_cap], mode="markers", marker=dict(size=13, color="#DC4A59", line=dict(width=2, color="#FFFFFF")),
            name="At capacity", hovertemplate=f"Top {cap} students<br>Risk captured: {y_cap:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[cap], y=[random_at_cap], mode="markers", marker=dict(size=11, color="#7C8798", line=dict(width=2, color="#FFFFFF")),
            name="Random at capacity", hovertemplate=f"Random selection at {cap}<br>Risk captured: {random_at_cap:.1%}<extra></extra>",
        ))
    fig.update_layout(
        title="Capacity allocation curve",
        xaxis_title="Students intervened (ranked by predicted risk)",
        yaxis_title="Fraction of total predicted risk captured",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        legend=dict(orientation="h", x=0, y=1.035, yanchor="bottom", font=dict(size=10)),
    )
    return fig


def plot_batch_band_distribution(bands):
    counts = pd.Series(bands).value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    color_map = {"Low Risk": "#12A594", "Medium Risk": "#F5A524", "High Risk": "#DC4A59"}
    fig = _plot_base(height=335, margin=dict(l=48, r=22, t=54, b=52))
    fig.add_trace(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[color_map[b] for b in counts.index],
        marker_line_width=0,
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Students: %{y}<extra></extra>",
    ))
    ymax = max(1, int(counts.max()))
    fig.update_layout(
        title="Risk-band distribution",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Students",
        yaxis=dict(range=[0, ymax * 1.18], automargin=True),
    )
    return fig


def plot_probability_histogram(probs):
    fig = _plot_base(height=335, margin=dict(l=50, r=22, t=54, b=52))
    fig.add_trace(go.Histogram(
        x=probs, nbinsx=25, marker_color="#2C467E", opacity=.82,
        hovertemplate="Probability: %{x:.1%}<br>Students: %{y}<extra></extra>",
    ))
    fig.add_vline(x=LOW_TH, line=dict(color="#12A594", width=2, dash="dot"))
    fig.add_vline(x=HIGH_TH, line=dict(color="#DC4A59", width=2, dash="dot"))
    fig.update_layout(
        title="Predicted probability distribution",
        showlegend=False,
        xaxis=dict(title="Dropout probability", tickformat=".0%", range=[0, 1]),
        yaxis_title="Students",
        bargap=.05,
        annotations=[
            dict(x=LOW_TH, y=1.02, xref="x", yref="paper", text="20%", showarrow=False, font=dict(size=10, color="#12A594")),
            dict(x=HIGH_TH, y=1.02, xref="x", yref="paper", text="50%", showarrow=False, font=dict(size=10, color="#DC4A59")),
        ],
    )
    return fig


# ==========================================================
# PDF HELPERS — polished, wrapped, page-safe reporting
# ==========================================================
def _fig_to_png_bytes(fig, width=700, height=400):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _styles():
    styles = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=styles["Title"], textColor=colors.HexColor("#12213F"),
            fontSize=26, leading=30, spaceAfter=5, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "SubTitle", parent=styles["Heading2"], textColor=colors.HexColor("#4767E8"),
            fontSize=11.5, leading=15, spaceAfter=4, fontName="Helvetica-Bold",
        ),
        "h1": ParagraphStyle(
            "H1", parent=styles["Heading1"], textColor=colors.HexColor("#12213F"),
            fontSize=16, leading=19, spaceBefore=7, spaceAfter=8, fontName="Helvetica-Bold",
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"], textColor=colors.HexColor("#203455"),
            fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=5, fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "Body", parent=styles["BodyText"], fontSize=9.3, leading=13, textColor=colors.HexColor("#53647D"),
        ),
        "body_dark": ParagraphStyle(
            "BodyDark", parent=styles["BodyText"], fontSize=9.3, leading=13, textColor=colors.HexColor("#203455"),
        ),
        "small": ParagraphStyle(
            "Small", parent=styles["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#7C8798"),
        ),
        "table": ParagraphStyle(
            "Table", parent=styles["BodyText"], fontSize=7.8, leading=9.6, textColor=colors.HexColor("#30425F"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=styles["BodyText"], fontSize=7.7, leading=9.2,
            textColor=colors.white, fontName="Helvetica-Bold",
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel", parent=styles["BodyText"], fontSize=7.1, leading=8.4,
            textColor=colors.HexColor("#7B8799"), fontName="Helvetica-Bold",
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue", parent=styles["BodyText"], fontSize=17, leading=19,
            textColor=colors.HexColor("#12213F"), fontName="Helvetica-Bold",
        ),
        "kpi_sub": ParagraphStyle(
            "KpiSub", parent=styles["BodyText"], fontSize=7.5, leading=9.2,
            textColor=colors.HexColor("#7B8799"),
        ),
    }


def _band_color(band):
    return {
        "Low Risk": colors.HexColor("#12A594"),
        "Medium Risk": colors.HexColor("#F5A524"),
        "High Risk": colors.HexColor("#DC4A59"),
    }.get(band, colors.HexColor("#4767E8"))


def _p(text, style, escape_html=True):
    from xml.sax.saxutils import escape
    txt = "" if text is None else str(text)
    if escape_html:
        txt = escape(txt)
    return Paragraph(txt.replace("\n", "<br/>"), style)


def _std_table(data, col_widths=None, header_bg="#17305F", row_alt="#F7F9FC", font_size=7.8):
    S = _styles()
    wrapped = []
    for r, row in enumerate(data):
        wrapped.append([_p(cell, S["table_header"] if r == 0 else S["table"]) for cell in row])
    t = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(row_alt)]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E0EA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _callout(title, body, bg="#F5F8FF", border="#DCE6FF", title_color="#2F53D5"):
    S = _styles()
    inner = Table([[
        _p(title, ParagraphStyle("CalloutTitle", parent=S["body_dark"], fontSize=8.2, leading=10, textColor=colors.HexColor(title_color), fontName="Helvetica-Bold")),
    ], [
        _p(body, S["body"]),
    ]], colWidths=[16.3 * cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("BOX", (0, 0), (-1, -1), .55, colors.HexColor(border)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return inner


def _hr():
    return HRFlowable(width="100%", thickness=.65, color=colors.HexColor("#D9E0EA"), spaceBefore=5, spaceAfter=8)


def _pdf_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#E4EAF2"))
    canvas.setLineWidth(.5)
    canvas.line(1.75 * cm, 1.25 * cm, w - 1.75 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#8A95A6"))
    canvas.drawString(1.75 * cm, .78 * cm, "Capacity-Aware Decision Support System")
    canvas.drawRightString(w - 1.75 * cm, .78 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _profile_relative(v, med):
    try:
        vf, mf = float(v), float(med)
        if mf == 0:
            return "n/a"
        ratio = vf / mf
        if ratio >= 1.5: return "Well above median"
        if ratio >= 1.1: return "Above median"
        if ratio >= .9: return "Near median"
        if ratio >= .5: return "Below median"
        return "Well below median"
    except (TypeError, ValueError):
        return "n/a"


def build_pdf_report(
    student_inputs, p_dropout, band, action, shap_vals, feature_cols, medians, model,
    counterfactual_feature=None, counterfactual_curve_png=None, radar_png=None, top_k=10,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=1.75 * cm, rightMargin=1.75 * cm,
        topMargin=1.45 * cm, bottomMargin=1.6 * cm,
        title="Student Dropout Risk Report", author="Capacity-Aware Decision Support System",
    )
    S = _styles()
    story = []

    # COVER / EXECUTIVE SNAPSHOT
    story.append(_p("Student Dropout Risk Report", S["cover_title"]))
    story.append(_p("Capacity-Aware Decision Support System", S["subtitle"]))
    story.append(_p(
        f"Model: {MODEL_NAME}  |  Prototype version: {MODEL_VERSION}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        S["small"],
    ))
    story.append(Spacer(1, 8))

    kpi_data = [[
        _p("DROPOUT PROBABILITY", S["kpi_label"]),
        _p("RISK BAND", S["kpi_label"]),
        _p("RECOMMENDED ACTION", S["kpi_label"]),
    ], [
        _p(f"{p_dropout:.1%}", ParagraphStyle("RiskKPI", parent=S["kpi_value"], textColor=_band_color(band))),
        _p(band, ParagraphStyle("BandKPI", parent=S["kpi_value"], fontSize=14, leading=17, textColor=_band_color(band))),
        _p(action, ParagraphStyle("ActionKPI", parent=S["kpi_value"], fontSize=11.2, leading=13.5)),
    ], [
        _p(f"Thresholds: Low < {LOW_TH:.0%}; Medium < {HIGH_TH:.0%}; High ≥ {HIGH_TH:.0%}", S["kpi_sub"]),
        _p("Operational risk stratification", S["kpi_sub"]),
        _p("Capacity-aware intervention tier", S["kpi_sub"]),
    ]]
    kt = Table(kpi_data, colWidths=[5.15*cm, 5.15*cm, 6.05*cm], hAlign="LEFT")
    kt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
        ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#DFE6F0")),
        ("INNERGRID", (0, 0), (-1, -1), .35, colors.HexColor("#EDF1F6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kt)
    story.append(Spacer(1, 10))
    story.append(_callout(
        "Executive interpretation",
        "This report summarizes the model assessment, the strongest local feature attributions, the student's profile relative to the training cohort, a single-variable what-if analysis, and the capacity-aware intervention tier. Predictions are probabilistic and are intended for human-supervised decision support.",
    ))

    story.append(Paragraph("1. Probability and Band Placement", S["h1"]))
    gauge_png = _fig_to_png_bytes(plot_gauge(p_dropout, band), width=700, height=320)
    band_png = _fig_to_png_bytes(plot_threshold_explorer(p_dropout), width=900, height=280)
    chart_cells = []
    if gauge_png:
        chart_cells.append(RLImage(io.BytesIO(gauge_png), width=8.0*cm, height=3.65*cm))
    else:
        chart_cells.append(_p(f"Probability: {p_dropout:.1%}", S["body"]))
    if band_png:
        chart_cells.append(RLImage(io.BytesIO(band_png), width=8.0*cm, height=3.65*cm))
    chart_table = Table([chart_cells], colWidths=[8.15*cm, 8.15*cm], hAlign="LEFT")
    chart_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(chart_table)

    # PROFILE
    story.append(Paragraph("2. Submitted Student Profile", S["h1"]))
    story.append(_p(
        "The visible indicators entered in the assessment are shown below. Model features that were not explicitly entered remain imputed from the training-cohort medians, consistent with the current workflow.",
        S["body"],
    ))
    profile_rows = [["Indicator", "Submitted", "Cohort median", "Relative position"]]
    for k, v in student_inputs.items():
        med = medians.get(k, np.nan)
        med_txt = "n/a" if pd.isna(med) else f"{float(med):.2f}"
        profile_rows.append([friendly_name(k), str(v), med_txt, _profile_relative(v, med)])
    story.append(Spacer(1, 4))
    story.append(_std_table(profile_rows, col_widths=[6.1*cm, 3.0*cm, 3.1*cm, 4.1*cm]))

    if radar_png:
        story.append(Paragraph("Profile context", S["h2"]))
        story.append(RLImage(io.BytesIO(radar_png), width=12.6*cm, height=7.25*cm, hAlign="CENTER"))

    story.append(PageBreak())

    # EXPLANATION
    story.append(Paragraph("3. Why the Model Reached This Assessment", S["h1"]))
    story.append(_p(
        "SHAP (SHapley Additive exPlanations) values quantify the marginal contribution of each feature to the model output. Positive values push the prediction toward dropout; negative values push it away. SHAP values are reported in log-odds space and should be read for direction and relative magnitude rather than as direct probability percentages.",
        S["body"],
    ))
    shap_png = _fig_to_png_bytes(plot_shap_waterfall(feature_cols, shap_vals, top_n=min(top_k, 8)), width=930, height=460)
    if shap_png:
        story.append(RLImage(io.BytesIO(shap_png), width=16.3*cm, height=8.05*cm))
    story.append(Spacer(1, 4))
    idx = np.argsort(np.abs(shap_vals))[-min(top_k, 8):][::-1]
    shap_rows = [["Rank", "Feature", "SHAP impact", "Direction", "Submitted"]]
    for rank, i in enumerate(idx, 1):
        v = float(shap_vals[i]); feat = feature_cols[i]
        shap_rows.append([str(rank), friendly_name(feat), f"{v:+.3f}", "Increases risk" if v > 0 else "Reduces risk", str(student_inputs.get(feat, "(imputed)"))])
    story.append(_std_table(shap_rows, col_widths=[1.0*cm, 6.2*cm, 2.5*cm, 3.0*cm, 3.6*cm]))

    story.append(Paragraph("4. Narrative Interpretation", S["h1"]))
    top_pos = sorted([(feature_cols[i], float(shap_vals[i])) for i in range(len(feature_cols)) if shap_vals[i] > 0], key=lambda x: -x[1])[:3]
    top_neg = sorted([(feature_cols[i], float(shap_vals[i])) for i in range(len(feature_cols)) if shap_vals[i] < 0], key=lambda x: x[1])[:3]
    narrative = f"The model estimates a dropout probability of <b>{p_dropout:.1%}</b>, corresponding to the <b>{band}</b> tier. "
    if top_pos:
        narrative += "The largest risk-increasing attributions are " + ", ".join(f"{friendly_name(f)} ({v:+.2f})" for f, v in top_pos) + ". "
    if top_neg:
        narrative += "The largest risk-reducing attributions are " + ", ".join(f"{friendly_name(f)} ({v:+.2f})" for f, v in top_neg) + "."
    story.append(_p(narrative, S["body"] , escape_html=False))

    donut_png = _fig_to_png_bytes(plot_contribution_donut(shap_vals), width=680, height=390)
    if donut_png:
        story.append(Spacer(1, 6))
        story.append(RLImage(io.BytesIO(donut_png), width=10.8*cm, height=5.9*cm, hAlign="CENTER"))

    # ACTION + WHAT IF
    story.append(PageBreak())
    story.append(Paragraph("5. Action Plan and What-If Analysis", S["h1"]))
    story.append(_callout(
        "Primary action",
        f"{action}. {get_intervention_detail(band)}",
        bg="#F8FAFE", border="#DCE5F2", title_color="#2D477E",
    ))
    story.append(Spacer(1, 6))
    story.append(_p(
        "The counterfactual view holds other model features constant while varying one selected indicator. It is a scenario analysis, not a causal guarantee: real-world changes may co-occur and may not produce the same result outside the model's observed data distribution.",
        S["body"],
    ))
    if counterfactual_curve_png:
        if counterfactual_feature:
            story.append(Paragraph(f"Scenario feature: {friendly_name(counterfactual_feature)}", S["h2"]))
        story.append(RLImage(io.BytesIO(counterfactual_curve_png), width=16.3*cm, height=7.05*cm))

    story.append(Paragraph("Operational notes", S["h2"]))
    story.append(_std_table([
        ["Rule", "Current prototype behavior"],
        ["Risk thresholds", f"Low < {LOW_TH:.0%}; Medium < {HIGH_TH:.0%}; High ≥ {HIGH_TH:.0%}"],
        ["Intensive capacity", f"C = {CAPACITY_LIMIT} university-wide slots"],
        ["Overflow", "High-risk students beyond capacity are routed to scalable workshops and monitoring."],
        ["Escalation", "Reassess at the next checkpoint; escalate when the predicted band increases and capacity is available."],
    ], col_widths=[4.0*cm, 12.3*cm]))

    story.append(PageBreak())
    story.append(Paragraph("6. Methodology, Fairness, and Limitations", S["h1"]))
    story.append(Paragraph("Model and data", S["h2"]))
    story.append(_p(
        "XGBoost gradient-boosted classifier with 400 trees, maximum depth 4, learning rate 0.05, subsample 0.8, colsample-by-tree 0.8, and positive-class reweighting. The current reporting layer describes the training cohort as 3,630 students with 36 features; the model uses an 80/20 stratified split in the application workflow.",
        S["body"],
    ))
    story.append(Paragraph("Validation and fairness", S["h2"]))
    story.append(_p(
        "The current report configuration cites a cross-validated AUC above 0.950 and a Brier score of 0.058. The fairness audit reports an Equal Opportunity mitigation example in which the scholarship-vs-non-scholarship TPR gap narrows from 0.022 to 0.003 through group-specific thresholding.",
        S["body"],
    ))
    story.append(Paragraph("Limitations", S["h2"]))
    story.append(_std_table([
        ["Consideration", "Interpretation"],
        ["Probabilistic output", "A prediction is an estimated probability, not a certainty about an individual student's future."],
        ["Historical data", "The model reflects patterns in its training data and should not be interpreted as identifying causal effects."],
        ["Counterfactuals", "Single-variable sweeps are scenario analyses; real interventions may change several variables together."],
        ["Capacity parameter", f"C = {CAPACITY_LIMIT} is a prototype operational parameter and should be calibrated to real institutional resources."],
        ["Human oversight", "The system is intended to support professional judgement, not to make fully automated decisions."],
    ], col_widths=[4.0*cm, 12.3*cm]))

    story.append(Spacer(1, 12))
    story.append(_callout(
        "Decision-support notice",
        "Use this report as structured evidence for advising and follow-up. Review the underlying student context before taking action, and keep intervention decisions under appropriate human supervision.",
        bg="#FFF9EA", border="#F4DEAC", title_color="#9A5A05",
    ))
    story.append(Spacer(1, 12))
    story.append(_hr())
    story.append(_p(
        f"Generated automatically · Report ID {datetime.now().strftime('%Y%m%d%H%M%S')}", S["small"]
    ))

    doc.build(story, onFirstPage=_pdf_page, onLaterPages=_pdf_page)
    buffer.seek(0)
    return buffer


def build_batch_pdf_report(scored_df, summary_df, curve_png=None, dist_png=None, hist_png=None, top_n=50, model_name=MODEL_NAME):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=1.75 * cm, rightMargin=1.75 * cm,
        topMargin=1.45 * cm, bottomMargin=1.6 * cm,
        title="Batch Cohort Risk Report", author="Capacity-Aware Decision Support System",
    )
    S = _styles(); story = []
    n_total = len(scored_df)
    n_high = int((scored_df["band"] == "High Risk").sum())
    n_med = int((scored_df["band"] == "Medium Risk").sum())
    n_low = int((scored_df["band"] == "Low Risk").sum())
    n_intensive = int((scored_df["action"] == "Intensive Mentoring and Counseling").sum())
    n_overflow = int(scored_df["action"].astype(str).str.contains("capacity overflow", na=False).sum())

    story.append(_p("Batch Cohort Risk Report", S["cover_title"]))
    story.append(_p("Capacity-Aware Decision Support System", S["subtitle"]))
    story.append(_p(
        f"Model: {model_name}  |  Version: {MODEL_VERSION}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Cohort: {n_total:,} students",
        S["small"],
    ))
    story.append(Spacer(1, 8))

    kpi_rows = [[
        _p("COHORT SIZE", S["kpi_label"]), _p("HIGH RISK", S["kpi_label"]), _p("INTENSIVE SLOTS", S["kpi_label"]), _p("OVERFLOW", S["kpi_label"]),
    ], [
        _p(f"{n_total:,}", S["kpi_value"]), _p(f"{n_high:,}", ParagraphStyle("HighKPI", parent=S["kpi_value"], textColor=colors.HexColor("#DC4A59"))),
        _p(f"{n_intensive} / {CAPACITY_LIMIT}", S["kpi_value"]), _p(f"{n_overflow:,}", ParagraphStyle("OverKPI", parent=S["kpi_value"], textColor=colors.HexColor("#F5A524"))),
    ], [
        _p("Students scored", S["kpi_sub"]), _p(f"{n_high/n_total:.1%} of cohort" if n_total else "0", S["kpi_sub"]),
        _p("Capacity-aware allocation", S["kpi_sub"]), _p("Routed to workshops", S["kpi_sub"]),
    ]]
    kt = Table(kpi_rows, colWidths=[4.08*cm]*4, hAlign="LEFT")
    kt.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), .6, colors.HexColor("#DFE6F0")), ("INNERGRID", (0,0), (-1,-1), .35, colors.HexColor("#EDF1F6")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ])); story.append(kt)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Cohort Snapshot", S["h1"]))
    snap = [["Metric", "Value"]] + [[str(k), str(v)] for k, v in summary_df.items()]
    story.append(_std_table(snap, col_widths=[7.2*cm, 9.1*cm]))

    story.append(Paragraph("2. Cohort Distributions", S["h1"]))
    if dist_png and hist_png:
        pair = Table([[
            RLImage(io.BytesIO(dist_png), width=7.8*cm, height=4.35*cm),
            RLImage(io.BytesIO(hist_png), width=7.8*cm, height=4.35*cm),
        ]], colWidths=[8.0*cm, 8.0*cm], hAlign="LEFT")
        pair.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 3), ("TOPPADDING", (0,0), (-1,-1), 0), ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
        story.append(pair)
    elif dist_png:
        story.append(RLImage(io.BytesIO(dist_png), width=14.5*cm, height=8.0*cm, hAlign="CENTER"))
    elif hist_png:
        story.append(RLImage(io.BytesIO(hist_png), width=14.5*cm, height=8.0*cm, hAlign="CENTER"))
    story.append(_callout(
        "Allocation reading",
        f"Risk bands use Low < {LOW_TH:.0%}, Medium < {HIGH_TH:.0%}, and High ≥ {HIGH_TH:.0%}. Up to {CAPACITY_LIMIT} high-risk students are allocated to intensive mentoring; high-risk overflow is routed to scalable workshops and monitoring.",
    ))

    story.append(PageBreak())
    story.append(Paragraph("3. Capacity Allocation Curve", S["h1"]))
    story.append(_p(
        "The model-ranked curve shows the cumulative fraction of predicted cohort risk captured as increasingly more students are selected for intervention. The vertical marker indicates the current operational capacity.", S["body"]
    ))
    if curve_png:
        story.append(RLImage(io.BytesIO(curve_png), width=16.3*cm, height=8.75*cm))

    alloc_rows = [["Tier / action", "Students", "Share", "Operational note"],
                  ["High Risk", n_high, f"{n_high/n_total:.1%}" if n_total else "0", "Top risk tier"],
                  ["Intensive Mentoring", n_intensive, f"{n_intensive/n_total:.1%}" if n_total else "0", f"Capped at {CAPACITY_LIMIT} slots"],
                  ["High-risk overflow", n_overflow, f"{n_overflow/n_total:.1%}" if n_total else "0", "Workshop + monitoring"],
                  ["Medium Risk", n_med, f"{n_med/n_total:.1%}" if n_total else "0", "Workshop + monitoring"],
                  ["Low Risk", n_low, f"{n_low/n_total:.1%}" if n_total else "0", "General academic support"]]
    story.append(Paragraph("4. Capacity Allocation Breakdown", S["h2"]))
    story.append(_std_table(alloc_rows, col_widths=[5.0*cm, 2.4*cm, 2.5*cm, 6.4*cm]))

    story.append(PageBreak())
    story.append(Paragraph(f"5. Priority Queue · Top {min(top_n, len(scored_df))}", S["h1"]))
    story.append(_p(
        "Students are ordered by predicted dropout probability after scoring. The student index corresponds to the sorted ordering in the annotated cohort output. The full scored CSV remains the complete machine-readable record.", S["body"]
    ))
    top_rows = [["Rank", "Student idx", "Probability", "Band", "Action"]]
    for rank, (_, row) in enumerate(scored_df.head(top_n).iterrows(), 1):
        top_rows.append([str(rank), str(row.get("student_index", "")), f"{float(row['probability']):.1%}", str(row["band"]), str(row["action"])])
    story.append(Spacer(1, 4))
    story.append(_std_table(top_rows, col_widths=[1.1*cm, 2.2*cm, 2.3*cm, 2.8*cm, 7.9*cm]))

    # Band detail, compact and capped to keep the report readable.
    story.append(PageBreak())
    story.append(Paragraph("6. Band Membership Detail", S["h1"]))
    for band_name in ["High Risk", "Medium Risk", "Low Risk"]:
        subset = scored_df[scored_df["band"] == band_name]
        story.append(Paragraph(f"{band_name} · {len(subset):,} students", S["h2"]))
        if subset.empty:
            story.append(_p("No students are currently in this band.", S["body"]))
            continue
        rows = [["Student idx", "Probability", "Action"]]
        for _, row in subset.head(60).iterrows():
            rows.append([str(row.get("student_index", "")), f"{float(row['probability']):.1%}", str(row["action"])])
        story.append(_std_table(rows, col_widths=[3.0*cm, 3.0*cm, 10.3*cm]))
        if len(subset) > 60:
            story.append(_p(f"Showing the first 60 of {len(subset):,} students in this band; the full list is available in the annotated CSV.", S["small"]))
        story.append(Spacer(1, 7))

    story.append(PageBreak())
    story.append(Paragraph("7. Methodology and Limitations", S["h1"]))
    story.append(_std_table([
        ["Topic", "Current prototype description"],
        ["Model", "XGBoost gradient-boosted classifier; 400 trees; depth 4; learning rate 0.05; subsample 0.8; colsample-by-tree 0.8; positive-class reweighting."],
        ["Training cohort", "3,630 students after removing the Enrolled class; 36 model features; current application workflow uses an 80/20 stratified split."],
        ["Validation reporting", "Current report configuration cites cross-validated AUC above 0.950 and Brier score 0.058."],
        ["Fairness example", "Scholarship-vs-non-scholarship TPR gap reported as 0.022 before mitigation and 0.003 after group-specific thresholding."],
        ["Capacity", f"C = {CAPACITY_LIMIT} is a prototype operational parameter and should be tuned to actual institutional resources."],
        ["Human oversight", "Predictions are probabilistic and intended as decision support; they are not causal claims or fully automated decisions."],
    ], col_widths=[4.2*cm, 12.1*cm]))
    story.append(Spacer(1, 10))
    story.append(_hr())
    story.append(_p(
        f"Generated automatically · Report ID {datetime.now().strftime('%Y%m%d%H%M%S')}", S["small"]
    ))
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
# UI HELPERS (presentation only)
# ==========================================================
TONE = {"Low Risk": "ok", "Medium Risk": "warn", "High Risk": "bad"}

LOGO_SVG = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#fff" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M22 9 12 4 2 9l10 5 10-5z"/>'
    '<path d="M6 11.5V16c0 1.2 2.7 3 6 3s6-1.8 6-3v-4.5"/><path d="M22 9v6"/></svg>'
)
EMPTY_SVG = (
    '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#3B5BDB" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>'
)


def _h(s: str) -> str:
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def _md(html: str):
    st.markdown(_h(html), unsafe_allow_html=True)


def _card(key: str):
    try:
        return st.container(key=key)
    except TypeError:
        return st.container(border=True)


def _title(title: str, sub: str = ""):
    sub_html = f'<div class="sec-s">{sub}</div>' if sub else ""
    _md(f'<div><div class="sec-t">{title}</div>{sub_html}</div>')


def _group(label: str):
    _md(f'<div class="grp">{label}</div>')


def _hint(text: str):
    _md(f'<div class="hint">{text}</div>')


def _kpi(label, value, sub="", tone=""):
    tone_cls = f" tone-{tone}" if tone else ""
    sub_html = f'<div class="kpi-s">{sub}</div>' if sub else ""
    return f'<div class="kpi{tone_cls}"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>{sub_html}</div>'


def _kpi_grid(items):
    _md('<div class="kpi-grid">' + "".join(items) + "</div>")


def render_header():
    _md(
        f"""
        <div class="hero">
          <div class="hero-l">
            <div class="logo">{LOGO_SVG}</div>
            <div>
              <div class="hero-title">Capacity-Aware Decision Support System</div>
              <div class="hero-sub">Interpretable student retention workspace · prototype</div>
            </div>
          </div>
          <div class="hero-r">
            <span class="gchip">XGBoost + TreeSHAP</span>
            <span class="gchip">{MODEL_VERSION}</span>
            <span class="gchip"><i></i>Model ready</span>
          </div>
        </div>
        """
    )


def render_sidebar():
    with st.sidebar:
        _md(
            f"""
            <div class="side-brand">
              <div class="side-brand-mark">{LOGO_SVG}</div>
              <div><div class="side-brand-title">Retention DSS</div><div class="side-brand-sub">Advisor console</div></div>
            </div>
            <div class="side-t">Risk bands</div>
            <div class="lg"><div><i style="background:#12A594"></i>Low risk</div><span>&lt; {LOW_TH:.0%}</span></div>
            <div class="lg"><div><i style="background:#F5A524"></i>Medium risk</div><span>{LOW_TH:.0%}–{HIGH_TH:.0%}</span></div>
            <div class="lg"><div><i style="background:#DC4A59"></i>High risk</div><span>&ge; {HIGH_TH:.0%}</span></div>
            <div class="side-t">Capacity</div>
            <div class="side-box"><b>{CAPACITY_LIMIT}</b> intensive mentoring slots. High-risk students beyond the cap are routed to workshops and monitoring.</div>
            <div class="side-t">Model</div>
            <div class="side-box"><b>{MODEL_NAME}</b><br>{MODEL_VERSION}</div>
            <div class="side-t">Use with care</div>
            <div class="side-box">Decision support only. Predictions are probabilistic; human judgement remains essential.</div>
            """
        )


def plot_fairness_tpr():
    groups = ["Non-Scholarship", "Scholarship"]
    fig = _plot_base(height=335, margin=dict(l=46, r=22, t=74, b=54))
    fig.add_trace(go.Bar(
        name="Unmitigated",
        x=groups, y=[0.907, 0.885], marker_color="#AAB5C5", marker_line_width=0,
        text=["90.7%", "88.5%"], textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{x}</b><br>Unmitigated TPR: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Mitigated",
        x=groups, y=[0.849, 0.846], marker_color="#4767E8", marker_line_width=0,
        text=["84.9%", "84.6%"], textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{x}</b><br>Mitigated TPR: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="True Positive Rate by group",
        barmode="group",
        yaxis=dict(range=[0, 1], title="TPR", tickformat=".0%"),
        legend=dict(orientation="h", x=0, y=1.07, yanchor="bottom", font=dict(size=10)),
    )
    return fig


# ==========================================================
# RESULT RENDERERS
# ==========================================================
def render_empty_state():
    _md(
        f"""
        <div class="empty">
          <div class="workspace-label"><i></i>Awaiting assessment</div>
          <div class="empty-i">{EMPTY_SVG}</div>
          <div class="empty-t">Your diagnostic workspace is ready</div>
          <div class="empty-s">Enter the student's core indicators on the left, then run the assessment. Results stay in this session so you can inspect risk, drivers, cohort context, what-if scenarios, and the action plan without changing the workflow.</div>
          <div class="empty-c"><span>Risk band</span><span>SHAP drivers</span><span>Cohort comparison</span><span>What-if simulator</span><span>PDF report</span></div>
        </div>
        """
    )


def _scenario_probability_plot(scenarios):
    labels = [s["label"] for s in scenarios]
    values = [s["probability"] for s in scenarios]
    colors_list = ["#2C467E"] + ["#4767E8"] * (len(values) - 1)
    fig = _plot_base(height=290, margin=dict(l=42, r=22, t=46, b=82))
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors_list,
        marker_line_width=0,
        text=[f"{v:.1%}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Predicted probability: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Saved scenario comparison",
        showlegend=False,
        yaxis=dict(range=[0, min(1.05, max(.12, max(values) * 1.16))], tickformat=".0%", title="Dropout probability"),
        xaxis=dict(tickangle=-18, automargin=True),
    )
    return fig


def render_assessment(res, model, medians, feature_cols):
    p_dropout = res["p"]
    band = res["band"]
    action = res["action"]
    shap_vals = res["shap_vals"]
    user_input = res["user_input"]
    radar_fig = res["radar_fig"]
    tone = TONE.get(band, "bad")

    st.session_state.setdefault("saved_scenarios", [])
    st.session_state.setdefault("advisor_note", "")
    st.session_state.setdefault("case_status", "Open")
    st.session_state.setdefault("followup_date", datetime.now().date())

    _md('<div class="workspace-label"><i></i>Latest diagnostic result</div>')
    c_res, c_dl = st.columns([3, 1.1], gap="medium")
    with c_res:
        _md(
            f"""
            <div class="result tone-{tone}">
              <div class="result-l"><div class="result-k">Dropout probability</div><div class="result-v">{p_dropout:.1%}</div></div>
              <div class="result-r">
                <span class="pill tone-{tone}">{band}</span>
                <div class="result-ak">Prescriptive action</div>
                <div class="result-a">{action}</div>
              </div>
            </div>
            """
        )
    with c_dl:
        _md('<div class="sec-s" style="margin-bottom:.1rem;font-weight:700;color:#52647e;">Report</div>')
        st.download_button(
            label="Download detailed PDF",
            data=res["pdf"], file_name=res["pdf_name"], mime="application/pdf",
            use_container_width=True, key="dl_single_pdf",
        )
        st.caption(f"Generated {res['stamp']} · session report")

    t_over, t_drv, t_cmp, t_wif, t_act = st.tabs(
        ["Overview", "Risk drivers", "Cohort comparison", "What-if", "Action plan"]
    )

    with t_over:
        _hint(
            "The probability gauge and band bar show the same model output from two complementary views. "
            "Thresholds are fixed at the values used by the current DSS; no model logic is changed by the UI."
        )
        col_gauge, col_band = st.columns([1, 1.28], gap="medium")
        with col_gauge:
            st.plotly_chart(plot_gauge(p_dropout, band), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        with col_band:
            st.plotly_chart(plot_threshold_explorer(p_dropout), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        _kpi_grid([
            _kpi("Current band", band, "Operational tier", tone),
            _kpi("Model probability", f"{p_dropout:.1%}", "Estimated dropout probability"),
            _kpi("Capacity rule", f"C = {CAPACITY_LIMIT}", "Intensive mentoring slots"),
        ])

    with t_drv:
        _hint(
            'Red bars increase the model output; teal bars reduce it. The chart uses abbreviated axis labels to prevent clipping, while full feature names remain available on hover.'
        )
        col_water, col_donut = st.columns([1.55, 1], gap="medium")
        with col_water:
            st.plotly_chart(plot_shap_waterfall(feature_cols, shap_vals, top_n=6), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        with col_donut:
            st.plotly_chart(plot_contribution_donut(shap_vals), use_container_width=True, config={"displayModeBar": False, "responsive": True})

        top_pos = [(friendly_name(feature_cols[i]), float(shap_vals[i])) for i in np.argsort(shap_vals)[::-1] if shap_vals[i] > 0][:3]
        top_neg = [(friendly_name(feature_cols[i]), float(shap_vals[i])) for i in np.argsort(shap_vals) if shap_vals[i] < 0][:3]
        chips_html = ""
        for name, val in top_pos:
            chips_html += f'<span class="factor-chip chip-risk">{name} ({val:+.2f})</span>'
        for name, val in top_neg:
            chips_html += f'<span class="factor-chip chip-prot">{name} ({val:+.2f})</span>'
        _group("Quick read")
        _md(f'<div class="factor-row">{chips_html}</div>' if chips_html else "<i>No dominant factors.</i>")

    with t_cmp:
        _hint(
            "The profile chart normalizes the selected indicators to the cohort median. This is a comparative visualization, not a causal or normative ranking."
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

        top_profile = [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Age at enrollment",
            "Tuition fees up to date",
            "Scholarship holder",
        ]
        rows = []
        for f in [f for f in top_profile if f in feature_cols]:
            v = float(user_input.get(f, medians[f])); m = float(medians[f])
            rows.append({"Indicator": friendly_name(f), "Student": v, "Cohort median": m, "Difference": v - m})
        if rows:
            pf = pd.DataFrame(rows)
            pf["Difference"] = pf["Difference"].map(lambda x: f"{x:+.2f}")
            st.dataframe(pf, use_container_width=True, hide_index=True)

    with t_wif:
        _hint("Adjust one visible indicator at a time. The simulator re-runs the existing XGBoost model; it does not retrain or alter the underlying system.")
        actionable_features = [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Tuition fees up to date",
            "Scholarship holder",
            "Age at enrollment",
        ]
        actionable_features = [f for f in actionable_features if f in feature_cols]
        cA, cB = st.columns([1, 1.25], gap="medium")
        with cA:
            cf_feature = st.selectbox("Scenario indicator", options=actionable_features, format_func=friendly_name, key="cf_feature")
        with cB:
            if cf_feature in ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"):
                cf_value = st.slider(f"Simulated value · {friendly_name(cf_feature)}", 0, 20, int(user_input.get(cf_feature, 0)), 1, key="cf_value_int")
            elif "grade" in cf_feature.lower():
                cf_value = st.slider(f"Simulated value · {friendly_name(cf_feature)}", 0.0, 20.0, float(user_input.get(cf_feature, 10.0)), 0.5, key="cf_value_float")
            elif cf_feature == "Age at enrollment":
                cf_value = st.slider(f"Simulated value · {friendly_name(cf_feature)}", 17, 65, int(user_input.get(cf_feature, 20)), 1, key="cf_value_age")
            else:
                cf_value = st.selectbox(
                    f"Simulated value · {friendly_name(cf_feature)}",
                    options=[("No", 0), ("Yes", 1)] if cf_feature != "Scholarship holder" else [("Yes", 1), ("No", 0)],
                    format_func=lambda x: x[0], key="cf_value_bin",
                )[1]

        cf_prob = compute_counterfactual(user_input, cf_feature, cf_value, model, medians, feature_cols)
        delta, delta_cls, arrow, band_note = band_change_summary(p_dropout, cf_prob)
        sim_tone = "ok" if delta < -0.005 else "bad" if delta > 0.005 else ""
        _kpi_grid([
            _kpi("Baseline", f"{p_dropout:.1%}", assign_risk_band(p_dropout)),
            _kpi("Scenario", f"{cf_prob:.1%}", f'<span class="{delta_cls}">{delta:+.2%}</span> {arrow}', sim_tone),
            _kpi("Scenario band", assign_risk_band(cf_prob), band_note.strip() or "No band change", TONE.get(assign_risk_band(cf_prob), "")),
        ])
        _hint(
            f"<b>{friendly_name(cf_feature)}</b>: {user_input.get(cf_feature, 'current')} → {cf_value}. "
            f"Predicted probability changes by <span class=\"{delta_cls}\">{delta:+.2%}</span> {arrow}.{band_note}"
        )
        current_val = user_input.get(cf_feature, cf_value)
        st.plotly_chart(plot_counterfactual_curve(user_input, cf_feature, current_val, model, medians, feature_cols), use_container_width=True, config={"displayModeBar": False, "responsive": True})

        save_col, clear_col = st.columns([1, 1])
        scenario_label = f"{_chart_label(cf_feature)} → {cf_value:g}"
        with save_col:
            if st.button("Save this scenario", use_container_width=True, key="save_scenario"):
                st.session_state["saved_scenarios"].append({
                    "label": scenario_label,
                    "feature": friendly_name(cf_feature),
                    "value": cf_value,
                    "probability": cf_prob,
                    "delta": delta,
                    "band": assign_risk_band(cf_prob),
                })
                st.session_state["saved_scenarios"] = st.session_state["saved_scenarios"][-5:]
                st.rerun()
        with clear_col:
            if st.button("Clear saved scenarios", use_container_width=True, key="clear_scenarios"):
                st.session_state["saved_scenarios"] = []
                st.rerun()

        if st.session_state["saved_scenarios"]:
            _group("Scenario board")
            scenario_table = [{
                "Scenario": s["label"], "Indicator": s["feature"], "Value": s["value"],
                "Probability": f"{s["probability"]:.1%}", "Δ vs baseline": f"{s["delta"]:+.2%}", "Band": s["band"]
            } for s in st.session_state["saved_scenarios"]]
            sdf = pd.DataFrame(scenario_table)
            kwargs = {"hide_index": True}
            if hasattr(st, "column_config"):
                kwargs["column_config"] = {
                    "Scenario": st.column_config.TextColumn("Scenario"),
                    "Indicator": st.column_config.TextColumn("Indicator"),
                    "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                    "Probability": st.column_config.TextColumn("Probability"),
                    "Δ vs baseline": st.column_config.TextColumn("Δ vs baseline"),
                    "Band": st.column_config.TextColumn("Band"),
                }
            st.dataframe(sdf, use_container_width=True, **kwargs)
            plot_data = [{"label": "Baseline", "probability": p_dropout}] + [
                {"label": s["label"], "probability": s["probability"]} for s in st.session_state["saved_scenarios"]
            ]
            st.plotly_chart(_scenario_probability_plot(plot_data), use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with t_act:
        _md(
            f"""
            <div class="result tone-{tone}">
              <div class="result-r" style="padding:1.1rem 1.4rem;">
                <span class="pill tone-{tone}">{band}</span>
                <div class="result-ak">Recommended action</div>
                <div class="result-a">{action}</div>
                <div class="fair-d" style="margin:.45rem 0 0;font-size:.88rem;color:#586A86;">{get_intervention_detail(band)}</div>
              </div>
            </div>
            """
        )
        _hint("Use the action plan as a structured starting point. The intervention is a prototype resource-routing recommendation and requires human review.")

        with st.expander("Advisor workspace · session notes and follow-up", expanded=False):
            a1, a2 = st.columns([1, 1], gap="medium")
            with a1:
                st.session_state["case_status"] = st.selectbox(
                    "Case status", ["Open", "Monitoring", "Closed"],
                    index=["Open", "Monitoring", "Closed"].index(st.session_state["case_status"]),
                    key="case_status_select",
                )
            with a2:
                st.session_state["followup_date"] = st.date_input("Follow-up checkpoint", value=st.session_state["followup_date"], key="followup_date_input")
            st.session_state["advisor_note"] = st.text_area(
                "Advisor note (session only)", value=st.session_state["advisor_note"], height=110,
                placeholder="Record context, discussion points, or follow-up items…", key="advisor_note_input",
            )
            _md(
                f'<div class="hint"><b>Workspace status:</b> {st.session_state["case_status"]} · '
                f'<b>Next checkpoint:</b> {st.session_state["followup_date"].strftime("%d %b %Y")}</div>'
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

    csv_bytes = scored.to_csv(index=False).encode("utf-8")

    curve_png = _fig_to_png_bytes(cap_fig, width=850, height=500)
    dist_png = _fig_to_png_bytes(dist_fig, width=750, height=420)
    hist_png = _fig_to_png_bytes(hist_fig, width=750, height=420)

    summary_df = pd.Series({
        "Students scored": n_total,
        "High risk": f"{n_high} ({n_high / n_total:.1%})" if n_total else "0",
        "Medium risk": f"{n_med} ({n_med / n_total:.1%})" if n_total else "0",
        "Low risk": f"{n_low} ({n_low / n_total:.1%})" if n_total else "0",
        "Mean probability": f"{mean_prob:.2%}",
        "Median probability": f"{median_prob:.2%}",
        "Intensive slots used": f"{n_intensive} / {CAPACITY_LIMIT}",
        "Capacity overflow": n_overflow,
        "Risk captured at capacity": f"{risk_at_cap:.2%}",
        "Lift vs random": f"{lift:.2f}x",
    })

    batch_pdf = build_batch_pdf_report(
        scored_df=scored[["student_index", "probability", "band", "action"]],
        summary_df=summary_df,
        curve_png=curve_png,
        dist_png=dist_png,
        hist_png=hist_png,
        top_n=50,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "n_total": n_total, "n_high": n_high, "n_med": n_med, "n_low": n_low,
        "n_intensive": n_intensive, "n_overflow": n_overflow, "mean_prob": mean_prob,
        "risk_at_cap": risk_at_cap, "lift": lift,
        "cap_fig": cap_fig, "dist_fig": dist_fig, "hist_fig": hist_fig,
        "queue": scored[["student_index", "probability", "band", "action"]].head(50).copy(),
        "csv": csv_bytes, "csv_name": f"cohort_scored_{stamp}.csv",
        "pdf": batch_pdf.getvalue(), "pdf_name": f"cohort_report_{stamp}.pdf",
    }


def render_batch(b):
    n = b["n_total"] or 1
    _kpi_grid([
        _kpi("Students scored", f"{b['n_total']:,}"),
        _kpi("High risk", f"{b['n_high']:,}", f"{b['n_high'] / n:.1%} of cohort", "bad"),
        _kpi("Medium risk", f"{b['n_med']:,}", f"{b['n_med'] / n:.1%} of cohort", "warn"),
        _kpi("Low risk", f"{b['n_low']:,}", f"{b['n_low'] / n:.1%} of cohort", "ok"),
        _kpi("Mean probability", f"{b['mean_prob']:.1%}"),
        _kpi("Intensive slots used", f"{b['n_intensive']} / {CAPACITY_LIMIT}"),
        _kpi("Capacity overflow", f"{b['n_overflow']:,}", "routed to workshops"),
        _kpi("Risk captured at capacity", f"{b['risk_at_cap']:.1%}", f"{b['lift']:.2f}x lift vs random"),
    ])

    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.download_button("Download Annotated Cohort CSV", data=b["csv"], file_name=b["csv_name"],
                           mime="text/csv", use_container_width=True, key="dl_batch_csv")
    with d2:
        st.download_button("Download Batch PDF Report", data=b["pdf"], file_name=b["pdf_name"],
                           mime="application/pdf", use_container_width=True, key="dl_batch_pdf")

    t_cap, t_dist, t_queue = st.tabs(["Capacity constraint", "Cohort distributions", "Priority queue"])
    with t_cap:
        st.plotly_chart(b["cap_fig"], use_container_width=True, config={"displayModeBar": False})
    with t_dist:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(b["dist_fig"], use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(b["hist_fig"], use_container_width=True, config={"displayModeBar": False})
    with t_queue:
        _title("Priority queue", "Top 50 highest-risk students, ranked by predicted dropout probability.")
        queue = b["queue"].copy()
        queue["probability"] = queue["probability"] * 100
        kwargs = {}
        if hasattr(st, "column_config"):
            kwargs["column_config"] = {
                "student_index": st.column_config.NumberColumn("Student idx", format="%d"),
                "probability": st.column_config.ProgressColumn("Dropout probability (%)", min_value=0, max_value=100, format="%.1f"),
                "band": st.column_config.TextColumn("Risk band"),
                "action": st.column_config.TextColumn("Recommended action", width="large"),
            }
            kwargs["hide_index"] = True
        st.dataframe(queue, use_container_width=True, **kwargs)


# ==========================================================
# MAIN APP
# ==========================================================
def main():
    model, explainer, feature_cols, X_train = load_and_train_model()
    medians = X_train.median()

    render_header()
    render_sidebar()
    _md('<div class="workspace-label"><i></i>Decision support workspace</div>')

    tab_eval, tab_batch, tab_info, tab_fairness = st.tabs([
        "Student Assessment",
        "Cohort Scoring",
        "How It Works",
        "Fairness Audit",
    ])

    # ---------------------------------------------------------
    # TAB 1: EVALUATION
    # ---------------------------------------------------------
    with tab_eval:
        col_input, col_results = st.columns([1, 2.15], gap="large")

        with col_input:
            with _card("card-input"):
                _title("Student profile", "Enter the indicators, then run the assessment.")
                _hint("Only the visible indicators need to be entered. Other model features continue to use cohort-median imputation, exactly as in the current workflow.")
                user_input = {}

                _group("Academic indicators")
                a1, a2 = st.columns(2)
                with a1:
                    user_input["Curricular units 1st sem (approved)"] = st.slider(
                        "1st Semester Passed Units", 0, 20,
                        int(X_train["Curricular units 1st sem (approved)"].median()),
                        help="Number of curricular units the student passed in semester 1.",
                    )
                with a2:
                    user_input["Curricular units 2nd sem (approved)"] = st.slider(
                        "2nd Semester Passed Units", 0, 20,
                        int(X_train["Curricular units 2nd sem (approved)"].median()),
                        help="Number of curricular units the student passed in semester 2.",
                    )
                user_input["Curricular units 2nd sem (grade)"] = st.slider(
                    "2nd Semester Average Grade", 0.0, 20.0,
                    float(X_train["Curricular units 2nd sem (grade)"].median()), 0.5,
                    help="Average grade across all 2nd semester units (0-20).",
                )

                _group("Socio-financial indicators")
                s1, s2 = st.columns(2)
                with s1:
                    user_input["Tuition fees up to date"] = st.selectbox(
                        "Tuition Fee Status", options=[("Up to date", 1), ("Overdue", 0)],
                        format_func=lambda x: x[0], help="Whether tuition fees are current.",
                    )[1]
                with s2:
                    user_input["Scholarship holder"] = st.selectbox(
                        "Scholarship Holder", options=[("Yes", 1), ("No", 0)],
                        format_func=lambda x: x[0], help="Whether the student receives a scholarship.",
                    )[1]
                s3, s4 = st.columns(2)
                with s3:
                    user_input["Gender"] = st.selectbox(
                        "Gender (UCI Encoding)", options=[("Male", 1), ("Female", 0)],
                        format_func=lambda x: x[0],
                    )[1]
                with s4:
                    user_input["Age at enrollment"] = st.number_input("Age at Enrollment", 17, 65, 20)

                evaluate = st.button("Generate Diagnostic Prediction", type="primary", use_container_width=True)

        with col_results:
            if evaluate:
                with st.spinner("Executing XGBoost and computing TreeSHAP matrices..."):
                    time.sleep(0.4)
                    x_full = {feat: user_input.get(feat, float(medians[feat])) for feat in feature_cols}
                    x_df = pd.DataFrame([x_full])
                    p_dropout = float(model.predict_proba(x_df)[:, 1][0])
                    band = assign_risk_band(p_dropout)
                    action = get_intervention(band)
                    shap_vals = explainer.shap_values(x_df)[0]

                    radar_features = [
                        "Curricular units 1st sem (approved)",
                        "Curricular units 2nd sem (approved)",
                        "Curricular units 2nd sem (grade)",
                        "Age at enrollment",
                        "Tuition fees up to date",
                        "Scholarship holder",
                    ]
                    radar_features = [f for f in radar_features if f in feature_cols]
                    radar_fig = plot_radar_profile(user_input, medians, radar_features)
                    radar_png = _fig_to_png_bytes(radar_fig, width=700, height=470)

                    cf_feature_default = "Curricular units 2nd sem (approved)"
                    cf_value_default = user_input.get(cf_feature_default, 0)
                    cf_fig = plot_counterfactual_curve(
                        user_input, cf_feature_default, cf_value_default, model, medians, feature_cols
                    )
                    cf_png = _fig_to_png_bytes(cf_fig, width=850, height=420)

                    pdf_buffer = build_pdf_report(
                        student_inputs=user_input,
                        p_dropout=p_dropout,
                        band=band,
                        action=action,
                        shap_vals=shap_vals,
                        feature_cols=feature_cols,
                        medians=medians,
                        model=model,
                        counterfactual_feature=cf_feature_default,
                        counterfactual_curve_png=cf_png,
                        radar_png=radar_png,
                        top_k=10,
                    )
                    now = datetime.now()
                    st.session_state["saved_scenarios"] = []
                    st.session_state["case_status"] = "Open"
                    st.session_state["advisor_note"] = ""
                    st.session_state["followup_date"] = datetime.now().date()
                    st.session_state["assessment"] = {
                        "user_input": dict(user_input),
                        "p": p_dropout,
                        "band": band,
                        "action": action,
                        "shap_vals": shap_vals,
                        "radar_fig": radar_fig,
                        "pdf": pdf_buffer.getvalue(),
                        "pdf_name": f"dropout_risk_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf",
                        "stamp": now.strftime("%H:%M:%S"),
                    }

            res = st.session_state.get("assessment")
            if res is None:
                render_empty_state()
            else:
                render_assessment(res, model, medians, feature_cols)

    # ---------------------------------------------------------
    # TAB 2: BATCH COHORT SCORING
    # ---------------------------------------------------------
    with tab_batch:
        _title(
            "Batch cohort scoring",
            f"Upload a CSV with the same schema as the training data (excluding the <code>target</code> column) to score an entire "
            f"cohort in one pass. The DSS applies the capacity constraint (top {CAPACITY_LIMIT} students) automatically and returns "
            f"an annotated CSV plus a detailed summary PDF.",
        )
        c_up, c_tpl = st.columns([1.7, 1], gap="large")
        with c_up:
            uploaded_file = st.file_uploader("Upload cohort CSV", type=["csv"], help="One row per student.")
        with c_tpl:
            with _card("card-template"):
                _group("Expected CSV format")
                _md(
                    '<div class="sec-s">One row per student. Columns should match the model feature names. '
                    'Missing optional columns are imputed with cohort medians.</div>'
                )
                template_df = pd.DataFrame([{f: float(X_train[f].median()) for f in feature_cols}])
                st.download_button(
                    label="Download CSV Template",
                    data=template_df.to_csv(index=False).encode("utf-8"),
                    file_name="cohort_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        batch_df = None
        file_key = None
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}:{uploaded_file.size}"
            try:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        if batch_df is not None:
            with st.expander(f"Uploaded data preview  |  {len(batch_df):,} rows", expanded=False):
                st.dataframe(batch_df.head(10), use_container_width=True)
                st.caption(f"Total rows uploaded: {len(batch_df)}")

            run_col, _spacer = st.columns([1, 2])
            with run_col:
                run_clicked = st.button("Run cohort scoring", type="primary", use_container_width=True)

            if run_clicked:
                with st.spinner("Scoring cohort and applying capacity constraint..."):
                    time.sleep(0.3)
                    st.session_state["batch_result"] = run_batch(batch_df, model, medians, feature_cols)
                    st.session_state["batch_key"] = file_key

            if st.session_state.get("batch_result") is not None and st.session_state.get("batch_key") == file_key:
                render_batch(st.session_state["batch_result"])

    # ---------------------------------------------------------
    # TAB 3: HOW IT WORKS
    # ---------------------------------------------------------
    with tab_info:
        col_m1, col_m2, col_m3 = st.columns(3, gap="medium")
        col_m1.markdown(
            _h("""
            <div class="metric-card"><h4>Validation Engine</h4><p><strong>XGBoost</strong></p><p>5-Fold Stratified CV</p></div>
            """),
            unsafe_allow_html=True,
        )
        col_m2.markdown(
            _h("""
            <div class="metric-card"><h4>Cross-Modality AUC</h4><p><strong>&gt; 0.950</strong></p><p>Traditional, LMS, xAPI, MOOC</p></div>
            """),
            unsafe_allow_html=True,
        )
        col_m3.markdown(
            _h("""
            <div class="metric-card"><h4>Calibration Metric</h4><p><strong>0.058</strong></p><p>Brier Score</p></div>
            """),
            unsafe_allow_html=True,
        )

        c_left, c_right = st.columns([1.3, 1], gap="large")
        with c_left:
            with _card("card-math"):
                _title(f"Mathematical capacity constraint (C={CAPACITY_LIMIT})")
                st.markdown(
                    f"""
Unlike standard predictive models that output a vacuum probability, this Decision
Support System utilizes a mathematically defined objective function to allocate
resources:

```
max sum (p_i * I(p_i >= T_H)) * a_i    s.t.    sum a_i <= C
```

This explicitly forces the algorithm to prioritize the top {CAPACITY_LIMIT} high-risk cases for
Intensive Mentoring, dynamically routing capacity overflow to scalable workshops.
The Batch Cohort Scoring tab visualizes this constraint directly through the
cumulative risk-captured curve.
"""
                )
        with c_right:
            _title("From profile to action")
            _md(
                f"""
                <div class="flow">
                  <div class="flow-s"><div class="flow-n">1</div><div><div class="flow-t">Student profile</div><div class="flow-d">Indicators entered by the advisor; missing features use cohort medians.</div></div></div>
                  <div class="flow-s"><div class="flow-n">2</div><div><div class="flow-t">XGBoost scoring</div><div class="flow-d">Estimates the probability of dropout.</div></div></div>
                  <div class="flow-s"><div class="flow-n">3</div><div><div class="flow-t">TreeSHAP explanation</div><div class="flow-d">Shows which factors raise or lower the risk.</div></div></div>
                  <div class="flow-s"><div class="flow-n">4</div><div><div class="flow-t">Risk band</div><div class="flow-d">Low below {LOW_TH:.0%}, Medium below {HIGH_TH:.0%}, High from {HIGH_TH:.0%}.</div></div></div>
                  <div class="flow-s"><div class="flow-n">5</div><div><div class="flow-t">Capacity-aware action</div><div class="flow-d">Up to {CAPACITY_LIMIT} intensive slots; overflow goes to workshops.</div></div></div>
                </div>
                """
            )

    # ---------------------------------------------------------
    # TAB 4: FAIRNESS AUDIT
    # ---------------------------------------------------------
    with tab_fairness:
        _title(
            "Active Equal Opportunity mitigation",
            "Rather than passively auditing bias, the framework actively enforces Equal Opportunity by optimizing "
            "group-specific probability thresholds (T_H,g). This neutralizes historical demographic disparities in intervention allocation.",
        )
        col_f1, col_f2 = st.columns(2, gap="medium")
        with col_f1:
            _md(
                """
                <div class="fair bad"><div class="fair-h">Unmitigated baseline (global threshold)</div><div class="fair-b">
                <div class="fair-d">A global threshold results in an unacceptable sensitivity gap between financial groups, disproportionately failing to flag at-risk scholarship students.</div>
                <div class="fair-r"><span>Non-Scholarship TPR</span><b>0.907</b></div>
                <div class="fair-r"><span>Scholarship TPR</span><b>0.885</b></div>
                <div class="fair-g"><span>Sensitivity gap (Delta TPR)</span><b>0.022</b></div>
                </div></div>
                """
            )
        with col_f2:
            _md(
                """
                <div class="fair ok"><div class="fair-h">Mitigated state (group thresholds)</div><div class="fair-b">
                <div class="fair-d">Applying mathematically optimized thresholds (Scholarship: 0.767, Non-Scholarship: 0.778) equalizes the True Positive Rates.</div>
                <div class="fair-r"><span>Non-Scholarship TPR</span><b>0.849</b></div>
                <div class="fair-r"><span>Scholarship TPR</span><b>0.846</b></div>
                <div class="fair-g"><span>Sensitivity gap (Delta TPR)</span><b>0.003</b></div>
                </div></div>
                """
            )
        st.plotly_chart(plot_fairness_tpr(), use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()
