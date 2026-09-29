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
from reportlab.platypus import CondPageBreak
from reportlab.pdfgen import canvas as rl_canvas
from xml.sax.saxutils import escape as xml_escape
import struct

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
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  color-scheme:light;
  --ink:#0E1B3D; --text:#4A5876; --muted:#7C89A6;
  --bg:#F4F6FB; --surface:#FFFFFF; --line:#E5EAF3; --line-2:#CFD8E8;
  --brand:#3B5BDB; --brand-d:#2B44B8; --brand-l:#EAF0FF; --brand-ll:#F5F8FF;
  --ok:#0B8A78; --ok-bg:#E8F8F4; --ok-line:#A6E3D6; --ok-s:#12A594;
  --warn:#9A5A05; --warn-bg:#FFF6E0; --warn-line:#F6D48A; --warn-s:#F5A524;
  --bad:#B42335; --bad-bg:#FEEEF0; --bad-line:#F5B5BD; --bad-s:#DC3F4A;
  --r:14px; --r-lg:18px;
  --sh1:0 1px 2px rgba(14,27,61,.05);
  --sh2:0 1px 2px rgba(14,27,61,.04), 0 8px 24px rgba(14,27,61,.06);
  --sh3:0 16px 40px rgba(14,27,61,.16);
  --ring:0 0 0 3px rgba(59,91,219,.25);
}

/* ---------- Base ---------- */
html, body, .stApp, .stApp *:not(code):not(pre):not([data-testid="stIconMaterial"]):not([class*="material"]){
  font-family:'Inter','Segoe UI',Roboto,Helvetica,Arial,sans-serif;
}
html{ -webkit-font-smoothing:antialiased; scroll-behavior:smooth; }
.stApp{ background:radial-gradient(900px 320px at 8% -6%, #E6EDFF 0%, rgba(230,237,255,0) 70%), var(--bg); color:var(--text); }
.block-container{ max-width:1320px; padding:2.6rem 2rem 4rem; animation:fadeUp .3s ease both; }
@keyframes fadeUp{ from{opacity:0; transform:translateY(5px);} to{opacity:1; transform:none;} }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent; height:2.4rem; }
div[data-testid="stVerticalBlock"]{ gap:.8rem; }
h1,h2,h3,h4,h5{ color:var(--ink); letter-spacing:-.015em; font-weight:700; }
h3{ font-size:1.05rem; margin:.4rem 0 .2rem; } h4{ font-size:.98rem; margin:.3rem 0 .2rem; }
p, li{ line-height:1.6; }
a{ color:var(--brand); text-decoration:none; } a:hover{ text-decoration:underline; }
hr{ border:0; height:1px; background:var(--line); margin:1rem 0; }
::selection{ background:var(--brand-l); color:var(--ink); }
*{ scrollbar-width:thin; scrollbar-color:#C4CEDF transparent; }
*::-webkit-scrollbar{ width:9px; height:9px; }
*::-webkit-scrollbar-thumb{ background:#C4CEDF; border-radius:99px; border:2px solid transparent; background-clip:content-box; }
div[data-testid="stElementContainer"]:has(> iframe[height="0"]), div[data-testid="element-container"]:has(> iframe[height="0"]), iframe[height="0"]{ display:none !important; }

/* ---------- Overflow safety ---------- */
.result-a,.kpi-v,.kpi-s,.kpi-l,.flow-t,.flow-d,.sec-t,.sec-s,.hint,.fair-d,.empty-s,.lg,.hero-title,.hero-sub,.lev-t,.lev-d,.side-box{ overflow-wrap:anywhere; word-break:normal; }
.result-r,.kpi,.lev-r > div,.flow-s > div{ min-width:0; }
label[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"]{ max-width:100%; }
label[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] > div{
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important; line-height:1.35; }
div[data-baseweb="select"] > div > div{ min-width:0; overflow:hidden; text-overflow:ellipsis; }
div[data-testid="stPlotlyChart"]{ overflow:hidden; max-width:100%; }
.stButton > button, .stDownloadButton > button{ height:auto; white-space:normal; line-height:1.25; }
.stButton > button p, .stDownloadButton > button p{ white-space:normal; }

/* ---------- Hero header ---------- */
.hero{ position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:linear-gradient(120deg,#16256B 0%,#2A45C6 55%,#4C6EF5 100%); color:#fff;
  border-radius:var(--r-lg); padding:1rem 1.3rem; margin:0 0 .9rem; box-shadow:0 10px 30px rgba(43,68,184,.28); }
.hero::after{ content:""; position:absolute; right:-70px; top:-90px; width:280px; height:280px; border-radius:50%;
  background:radial-gradient(circle, rgba(255,255,255,.2), rgba(255,255,255,0) 70%); pointer-events:none; }
.hero-l{ display:flex; align-items:center; gap:.85rem; position:relative; z-index:1; min-width:0; }
.logo{ width:44px; height:44px; border-radius:13px; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
  display:flex; align-items:center; justify-content:center; flex:none; }
.hero-title{ font-size:1.18rem; font-weight:750; letter-spacing:-.015em; line-height:1.2; color:#fff; }
.hero-sub{ font-size:.82rem; color:rgba(255,255,255,.8); margin-top:2px; }
.hero-r{ display:flex; gap:.4rem; flex-wrap:wrap; position:relative; z-index:1; }
.gchip{ display:inline-flex; align-items:center; gap:.4rem; font-size:.73rem; font-weight:600; color:#fff; padding:.28rem .7rem;
  border-radius:99px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26); white-space:nowrap; }
.gchip i{ width:7px; height:7px; border-radius:50%; background:#5EEAD4; box-shadow:0 0 0 3px rgba(94,234,212,.28); }

/* ---------- Top-level tabs = segmented nav ---------- */
[data-baseweb="tab-list"]{ gap:.25rem; padding:.3rem; background:var(--surface); border:1px solid var(--line); border-radius:14px;
  box-shadow:var(--sh1); position:sticky; top:2.6rem; z-index:30; margin-bottom:.9rem; overflow-x:auto; scrollbar-width:none; }
[data-baseweb="tab-list"]::-webkit-scrollbar{ display:none; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none !important; }
button[role="tab"]{ height:auto; padding:.55rem 1.05rem; border-radius:10px; background:transparent; color:var(--muted);
  font-weight:600; font-size:.9rem; transition:background .15s, color .15s; white-space:nowrap; flex:none; }
button[role="tab"] p{ font-size:.9rem; font-weight:600; margin:0; color:inherit; }
button[role="tab"]:hover{ background:var(--brand-ll); color:var(--brand-d); }
button[role="tab"][aria-selected="true"]{ background:linear-gradient(180deg,#4A6BE8,#3B5BDB); color:#fff; box-shadow:0 4px 12px rgba(59,91,219,.32); }
button[role="tab"][aria-selected="true"] p{ color:#fff; }
div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab-list"]{
  position:static; background:transparent; border:0; border-bottom:1px solid var(--line); border-radius:0; box-shadow:none; padding:0; gap:.15rem; margin-bottom:.8rem; }
div[data-testid="stTabs"] div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ display:block !important; background:var(--brand); height:3px; border-radius:3px 3px 0 0; }
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"]{ border-radius:8px 8px 0 0; padding:.6rem .9rem; font-size:.88rem; }
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{ background:transparent; color:var(--brand-d); box-shadow:none; }
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{ color:var(--brand-d); }

/* ---------- Cards (keyed containers) ---------- */
div[class*="st-key-card"]{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--sh2); padding:1.05rem 1.15rem 1.15rem; }
div[class*="st-key-card"] div[data-testid="stVerticalBlock"]{ gap:.65rem; }
.sec-t{ font-size:1.02rem; font-weight:700; color:var(--ink); letter-spacing:-.01em; }
.sec-s{ font-size:.83rem; color:var(--muted); margin-top:2px; line-height:1.5; }
.grp{ display:flex; align-items:center; gap:.6rem; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:.35rem 0 .1rem; }
.grp::after{ content:""; flex:1; height:1px; background:var(--line); }
.hint{ background:var(--brand-ll); border:1px solid #DCE6FF; border-radius:12px; padding:.6rem .85rem; font-size:.86rem; color:var(--text); line-height:1.55; }
.hint b{ color:var(--ink); }

/* ---------- Result hero card ---------- */
.result{ display:flex; align-items:stretch; background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--sh2); overflow:hidden; }
.result-l{ flex:none; min-width:170px; padding:.9rem 1.3rem; display:flex; flex-direction:column; justify-content:center; color:#fff; }
.result.tone-ok .result-l{ background:linear-gradient(135deg,#0E9F8B,#12A594); }
.result.tone-warn .result-l{ background:linear-gradient(135deg,#E8930C,#F5A524); }
.result.tone-bad .result-l{ background:linear-gradient(135deg,#C92F3C,#DC3F4A); }
.result-k{ font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; opacity:.92; }
.result-v{ font-size:2.5rem; font-weight:800; letter-spacing:-.03em; line-height:1.05; margin-top:.15rem; }
.result-r{ flex:1; padding:.9rem 1.3rem; display:flex; flex-direction:column; justify-content:center; gap:.35rem; }
.pill{ display:inline-flex; align-self:flex-start; align-items:center; gap:.4rem; font-size:.72rem; font-weight:750; text-transform:uppercase; letter-spacing:.08em; padding:.22rem .65rem; border-radius:99px; border:1px solid; }
.pill::before{ content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }
.pill.tone-ok{ background:var(--ok-bg); color:var(--ok); border-color:var(--ok-line); }
.pill.tone-warn{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.pill.tone-bad{ background:var(--bad-bg); color:var(--bad); border-color:var(--bad-line); }
.result-ak{ font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); margin-top:.15rem; }
.result-a{ font-size:1.08rem; font-weight:700; color:var(--ink); letter-spacing:-.01em; }

/* ---------- KPI tiles ---------- */
.kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:.7rem; }
.kpi{ position:relative; overflow:hidden; background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:.8rem 1rem .85rem 1.1rem; box-shadow:var(--sh1); }
.kpi::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--brand); }
.kpi.tone-ok::before{ background:var(--ok-s); } .kpi.tone-warn::before{ background:var(--warn-s); } .kpi.tone-bad::before{ background:var(--bad-s); }
.kpi-l{ font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }
.kpi-v{ font-size:clamp(1.15rem,1.5vw,1.5rem); font-weight:800; color:var(--ink); letter-spacing:-.02em; line-height:1.2; margin-top:.15rem; }
.kpi-s{ font-size:.78rem; color:var(--muted); margin-top:.1rem; }

/* ---------- Levers ---------- */
.lev{ display:flex; flex-direction:column; gap:.5rem; }
.lev-r{ display:grid; grid-template-columns:28px minmax(0,1fr) minmax(90px,150px); gap:.75rem; align-items:center; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.65rem .85rem; box-shadow:var(--sh1); }
.lev-n{ width:26px; height:26px; border-radius:50%; background:var(--ok-bg); color:var(--ok); font-weight:800; font-size:.78rem; display:flex; align-items:center; justify-content:center; }
.lev-t{ font-weight:650; color:var(--ink); font-size:.9rem; }
.lev-d{ font-size:.8rem; color:var(--muted); }
.lev-v{ text-align:right; } .lev-v b{ color:var(--ok); font-size:1.05rem; }
.lev-track{ height:6px; border-radius:99px; background:var(--line); margin-top:.35rem; overflow:hidden; }
.lev-fill{ height:100%; border-radius:99px; background:linear-gradient(90deg,#12A594,#5EEAD4); }

/* ---------- Empty state ---------- */
.empty{ text-align:center; padding:3rem 1.5rem; background:var(--surface); border:1.5px dashed var(--line-2); border-radius:var(--r-lg); }
.empty-i{ width:56px; height:56px; margin:0 auto .9rem; border-radius:16px; background:var(--brand-l); display:flex; align-items:center; justify-content:center; }
.empty-t{ font-size:1.05rem; font-weight:700; color:var(--ink); }
.empty-s{ max-width:520px; margin:.35rem auto 0; font-size:.88rem; color:var(--muted); line-height:1.6; }
.empty-c{ display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin-top:1rem; }
.empty-c span{ font-size:.75rem; font-weight:600; color:var(--brand-d); background:var(--brand-l); padding:.28rem .7rem; border-radius:99px; }

/* ---------- Info / fairness / sidebar blocks ---------- */
.metric-card{ position:relative; overflow:hidden; background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:1rem 1.15rem; box-shadow:var(--sh2); height:100%; }
.metric-card::before{ content:""; position:absolute; left:0; right:0; top:0; height:3px; background:linear-gradient(90deg,var(--brand),#8DA6FF); }
.metric-card h4{ margin:.15rem 0 .35rem; font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:700; }
.metric-card p{ margin:.05rem 0; color:var(--text); font-size:.86rem; }
.metric-card p strong{ color:var(--ink); font-size:1.5rem; font-weight:800; letter-spacing:-.02em; }
.flow{ display:flex; flex-direction:column; gap:.5rem; }
.flow-s{ display:flex; gap:.75rem; align-items:flex-start; padding:.65rem .8rem; border:1px solid var(--line); border-radius:12px; background:var(--surface); }
.flow-n{ flex:none; width:26px; height:26px; border-radius:50%; background:var(--brand-l); color:var(--brand-d); font-weight:800; font-size:.78rem; display:flex; align-items:center; justify-content:center; }
.flow-t{ font-weight:650; color:var(--ink); font-size:.9rem; } .flow-d{ font-size:.8rem; color:var(--muted); line-height:1.5; }
.fair{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--sh2); overflow:hidden; height:100%; }
.fair-h{ padding:.7rem 1.1rem; font-size:.7rem; font-weight:750; text-transform:uppercase; letter-spacing:.1em; }
.fair.bad .fair-h{ background:var(--bad-bg); color:var(--bad); } .fair.ok .fair-h{ background:var(--ok-bg); color:var(--ok); }
.fair-b{ padding:.6rem 1.1rem 1rem; }
.fair-d{ font-size:.84rem; color:var(--muted); line-height:1.55; margin:.2rem 0 .5rem; }
.fair-r{ display:flex; justify-content:space-between; align-items:baseline; gap:.75rem; padding:.5rem 0; border-top:1px solid var(--line); font-size:.88rem; }
.fair-r b{ color:var(--ink); font-size:1rem; }
.fair-g{ display:flex; justify-content:space-between; align-items:baseline; gap:.75rem; padding:.65rem .8rem; margin-top:.4rem; border-radius:12px; font-weight:650; font-size:.88rem; }
.fair.bad .fair-g{ background:var(--bad-bg); color:var(--bad); } .fair.ok .fair-g{ background:var(--ok-bg); color:var(--ok); }
.fair-g b{ font-size:1.35rem; font-weight:800; }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{ padding:1rem 1rem 2rem; }
.side-t{ font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:1rem 0 .5rem; }
.side-t:first-child{ margin-top:.2rem; }
.lg{ display:flex; align-items:center; justify-content:space-between; gap:.5rem; padding:.5rem .7rem; border:1px solid var(--line); border-radius:10px; margin-bottom:.35rem; font-size:.85rem; color:var(--ink); font-weight:600; }
.lg span{ font-weight:500; color:var(--muted); font-size:.8rem; white-space:nowrap; }
.lg i{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.5rem; }
.side-box{ background:var(--brand-ll); border:1px solid #DCE6FF; border-radius:12px; padding:.65rem .8rem; font-size:.82rem; color:var(--text); line-height:1.55; }
.side-box b{ color:var(--ink); }

/* ---------- Form controls ---------- */
label[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] p{ font-size:.8rem; font-weight:600; color:var(--ink); }
div[data-testid="stTooltipContent"]{ background:var(--ink); color:#fff; border-radius:8px; font-size:.8rem; box-shadow:var(--sh3); }
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div{
  background:var(--surface) !important; border:1px solid var(--line-2) !important; border-radius:10px !important; box-shadow:var(--sh1); min-height:40px; transition:border-color .15s, box-shadow .15s; }
div[data-baseweb="input"]:hover, div[data-baseweb="select"] > div:hover{ border-color:#9FB1CE !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within{ border-color:var(--brand) !important; box-shadow:var(--ring) !important; }
div[data-baseweb="input"] input, div[data-baseweb="select"] input, textarea{ color:var(--ink) !important; -webkit-text-fill-color:var(--ink); font-size:.9rem; font-weight:500; }
div[data-testid="stNumberInput"] div[data-baseweb="input"]{ overflow:hidden; }
div[data-testid="stNumberInput"] button{ background:var(--brand-ll) !important; color:var(--brand-d) !important; border:0 !important; border-left:1px solid var(--line) !important; }
div[data-testid="stNumberInput"] button:hover{ background:var(--brand-l) !important; }
div[data-baseweb="select"] svg{ color:var(--muted); }
span[data-baseweb="tag"]{ background:var(--brand-l) !important; color:var(--brand-d) !important; border-radius:6px !important; font-weight:600; }
div[data-baseweb="popover"] > div{ background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:12px !important; box-shadow:var(--sh3) !important; overflow:hidden; }
div[data-baseweb="popover"] ul, ul[role="listbox"]{ background:var(--surface) !important; padding:.3rem !important; }
div[data-baseweb="popover"] li, li[role="option"]{ border-radius:8px !important; margin:1px 0; padding:.5rem .7rem !important; color:var(--ink) !important; font-size:.9rem; font-weight:500; background:transparent !important; }
div[data-baseweb="popover"] li:hover, li[role="option"]:hover{ background:var(--brand-ll) !important; color:var(--brand-d) !important; }
li[role="option"][aria-selected="true"]{ background:var(--brand-l) !important; color:var(--brand-d) !important; font-weight:650; }
div[data-testid="stSlider"]{ padding:0 .3rem .1rem; }
div[data-testid="stSlider"] [role="slider"]{ background:var(--brand) !important; border:3px solid #fff !important; box-shadow:0 0 0 1px var(--brand), var(--sh2) !important; height:1.1rem; width:1.1rem; }
div[data-testid="stSliderThumbValue"]{ color:var(--brand-d) !important; font-weight:700; font-size:.82rem; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"]{ color:var(--muted); font-size:.72rem; }
input[type="checkbox"], input[type="radio"]{ accent-color:var(--brand); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  min-height:42px; padding:.5rem 1.05rem; border-radius:11px; font-weight:650; font-size:.9rem; background:var(--surface); color:var(--brand-d);
  border:1px solid var(--line-2); box-shadow:var(--sh1); transition:transform .12s, box-shadow .15s, background .15s, border-color .15s; }
.stButton > button p, .stDownloadButton > button p{ font-weight:650; font-size:.9rem; margin:0; }
.stButton > button:hover, .stDownloadButton > button:hover{ background:var(--brand-ll); border-color:var(--brand); color:var(--brand-d); box-shadow:var(--sh2); }
.stButton > button:active, .stDownloadButton > button:active{ transform:translateY(1px); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, button[role="tab"]:focus-visible, summary:focus-visible{ outline:none; box-shadow:var(--ring); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(180deg,#4A6BE8,#3B5BDB); color:#fff; border-color:#3554C9; box-shadow:0 1px 2px rgba(43,68,184,.4), 0 8px 18px rgba(59,91,219,.28); }
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p{ color:#fff; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{ background:linear-gradient(180deg,#5776F0,#3F60E2); color:#fff; border-color:#3554C9; }

/* ---------- Uploader / expander / charts / tables / alerts ---------- */
div[data-testid="stFileUploader"] section{ background:var(--surface); border:1.5px dashed #A9B9D6; border-radius:var(--r-lg); padding:1.2rem; transition:all .15s; }
div[data-testid="stFileUploader"] section:hover{ border-color:var(--brand); background:var(--brand-ll); }
div[data-testid="stFileUploader"] small{ color:var(--muted); }
div[data-testid="stFileUploaderFile"]{ background:var(--brand-ll); border-radius:10px; }
div[data-testid="stExpander"]{ background:var(--surface); border:1px solid var(--line) !important; border-radius:var(--r) !important; box-shadow:var(--sh1); overflow:hidden; }
div[data-testid="stExpander"] details{ border:0 !important; }
div[data-testid="stExpander"] summary{ padding:.7rem 1rem; font-weight:650; color:var(--ink); }
div[data-testid="stExpander"] summary:hover{ background:var(--brand-ll); }
div[data-testid="stExpander"] summary p{ font-size:.92rem; font-weight:650; margin:0; }
div[data-testid="stPlotlyChart"]{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r); padding:.4rem .5rem .1rem; box-shadow:var(--sh1); }
div[data-testid="stDataFrame"], div[data-testid="stTable"]{ border:1px solid var(--line); border-radius:var(--r); overflow:hidden; box-shadow:var(--sh1); }
div[data-testid="stAlert"]{ border-radius:var(--r); border:1px solid var(--line); }
div[data-testid="stCaptionContainer"], .stCaption{ color:var(--muted); font-size:.8rem; }
div[data-testid="stSpinner"] p{ color:var(--muted); font-weight:500; }
.stMarkdown code{ background:var(--brand-l); color:var(--brand-d); border-radius:6px; padding:.1rem .4rem; font-size:.85em; }
div[data-testid="stCode"] pre, pre{ background:#0E1B3D !important; color:#E6EDFF !important; border-radius:12px; font-size:.85rem; overflow-x:auto; }
.delta-good{ color:var(--ok); font-weight:700; } .delta-bad{ color:var(--bad); font-weight:700; } .delta-flat{ color:var(--muted); font-weight:700; }

@media (max-width:900px){
  .block-container{ padding:2.4rem .9rem 3rem; }
  .result{ flex-direction:column; } .result-l{ min-width:0; }
  .lev-r{ grid-template-columns:28px minmax(0,1fr); } .lev-v{ grid-column:1 / -1; text-align:left; }
  button[role="tab"]{ padding:.5rem .7rem; font-size:.82rem; }
}
@media (prefers-reduced-motion:reduce){ *{ animation:none !important; transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

components.html(
    r"""
<script>
(function () {
  try {
    const doc = window.parent.document;
    if (doc.getElementById("campus-enhancer")) return;
    const mark = doc.createElement("meta"); mark.id = "campus-enhancer"; mark.name = "theme-color"; mark.content = "#3B5BDB";
    doc.head.appendChild(mark);
    const link = doc.createElement("link"); link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap";
    doc.head.appendChild(link);
    doc.documentElement.setAttribute("lang", "en");

    const BRAND = "rgb(59, 91, 219)";
    const isRed = s => /rgb\(255,\s*75,\s*75\)|#ff4b4b/i.test(s);
    const fixRed = s => s.replace(/rgb\(255,\s*75,\s*75\)|#ff4b4b/gi, BRAND);
    const enhance = () => {
      doc.querySelectorAll('[data-baseweb="slider"] *').forEach(el => {
        const s = el.getAttribute("style");
        if (s && isRed(s)) el.setAttribute("style", fixRed(s));
      });
      doc.querySelectorAll('button[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] button')
        .forEach(b => { if (!b.getAttribute("aria-label")) b.setAttribute("aria-label", "Toggle sidebar"); });
      doc.querySelectorAll('div[data-testid="stPlotlyChart"]').forEach(c => {
        if (!c.getAttribute("aria-label")) c.setAttribute("aria-label", "Interactive chart");
      });
    };
    enhance();
    let t = null;
    new MutationObserver(() => { clearTimeout(t); t = setTimeout(enhance, 80); })
      .observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["style"] });
  } catch (e) { /* CSS still works if this fails */ }
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
_C_INK = "#0E1B3D"
_C_MUTED = "#7C89A6"
_C_BRAND = "#3B5BDB"
_C_NAVY = "#1B2A5C"
_C_OK = "#12A594"
_C_WARN = "#F5A524"
_C_BAD = "#DC3F4A"


def _band_hex(band):
    return _C_OK if band == "Low Risk" else _C_WARN if band == "Medium Risk" else _C_BAD


def _ttl(text, sub=None):
    t = text if not sub else f"{text}<br><sup>{sub}</sup>"
    return dict(text=t, x=0, xanchor="left", y=0.98, yanchor="top", yref="container",
                font=dict(size=14, color=_C_INK), automargin=True)


def _legend_bottom():
    return dict(orientation="h", yanchor="bottom", y=0, yref="container", xanchor="center", x=0.5,
                font=dict(size=11.5), bgcolor="rgba(0,0,0,0)")


def _transparent():
    return dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")


def plot_gauge(probability, band):
    color = _band_hex(band)
    lo, hi = LOW_TH * 100, HIGH_TH * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 38, "color": color}},
            title={"text": "Dropout probability", "font": {"size": 13, "color": _C_MUTED}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, lo, hi, 100], "ticktext": ["0", f"{lo:.0f}", f"{hi:.0f}", "100"],
                         "tickwidth": 1, "tickcolor": "#94A3B8", "tickfont": {"size": 11}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, lo], "color": "rgba(18, 165, 148, 0.20)"},
                    {"range": [lo, hi], "color": "rgba(245, 165, 36, 0.22)"},
                    {"range": [hi, 100], "color": "rgba(220, 63, 74, 0.20)"},
                ],
                "threshold": {"line": {"color": _C_INK, "width": 3}, "thickness": 0.85, "value": probability * 100},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=34, r=34, t=70, b=20), **_transparent())
    return fig


def plot_shap_waterfall(feature_names, shap_values, top_n=6):
    idx = np.argsort(np.abs(shap_values))[-top_n:]
    names = [friendly_name(feature_names[i]) for i in idx]
    vals = np.asarray(shap_values)[idx]
    lim = max(float(np.max(np.abs(vals))), 1e-6) * 1.4
    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=[_C_BAD if v > 0 else _C_OK for v in vals],
            text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
            textfont=dict(size=11.5, color=_C_INK),
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=_ttl("Top factors driving risk", "SHAP attributions (log-odds)"),
        xaxis=dict(title="Protects (green)  |  Increases risk (red)", range=[-lim, lim], zeroline=True,
                   zerolinecolor=_C_NAVY, zerolinewidth=2, showgrid=True, automargin=True),
        yaxis=dict(title="", automargin=True, tickfont=dict(size=12)),
        height=max(310, 120 + 42 * len(vals)),
        margin=dict(l=10, r=30, t=76, b=60),
        showlegend=False,
        **_transparent(),
    )
    return fig


def plot_contribution_donut(shap_values):
    pos = float(np.sum(shap_values[shap_values > 0]))
    neg = float(np.sum(np.abs(shap_values[shap_values < 0])))
    fig = go.Figure(
        go.Pie(
            labels=["Risk-increasing factors", "Risk-reducing factors"],
            values=[pos, neg], hole=0.62, sort=False, direction="clockwise",
            marker=dict(colors=[_C_BAD, _C_OK], line=dict(color="white", width=3)),
            textinfo="percent", textposition="inside", insidetextorientation="horizontal",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Total impact: %{value:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=_ttl("Balance of contributions", "Share of total SHAP impact"),
        height=330, margin=dict(l=20, r=20, t=76, b=64),
        showlegend=True, legend=_legend_bottom(),
        annotations=[dict(text="Risk<br>balance", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color=_C_MUTED))],
        **_transparent(),
    )
    return fig


def plot_threshold_explorer(probability):
    edges = [(0.0, LOW_TH, "Low", _C_OK), (LOW_TH, HIGH_TH, "Medium", _C_WARN), (HIGH_TH, 1.0, "High", _C_BAD)]
    fig = go.Figure()
    fills = {"Low": "rgba(18, 165, 148, 0.30)", "Medium": "rgba(245, 165, 36, 0.34)", "High": "rgba(220, 63, 74, 0.30)"}
    for a, b, name, col in edges:
        fig.add_trace(go.Bar(
            x=[b - a], base=[a], y=[0], orientation="h", width=0.5, marker=dict(color=fills[name], line=dict(color="white", width=2)),
            text=[name], textposition="inside", insidetextanchor="middle", textfont=dict(size=12, color=col),
            hovertemplate=f"{name} risk<br>{a:.2f} - {b:.2f}<extra></extra>", showlegend=False,
        ))
    fig.add_shape(type="line", x0=probability, x1=probability, y0=-0.32, y1=0.32, line=dict(color=_C_INK, width=4))
    anchor = "left" if probability < 0.18 else "right" if probability > 0.82 else "center"
    fig.add_annotation(
        x=probability, y=0.32, ax=0, ay=-34, xanchor=anchor, showarrow=True, arrowhead=2, arrowcolor=_C_INK,
        text=f"<b>This student: {probability:.1%}</b>", font=dict(size=12.5, color=_C_INK),
        bgcolor="white", bordercolor=_C_INK, borderwidth=1, borderpad=4,
    )
    fig.update_layout(
        barmode="overlay",
        title=_ttl("Where this student falls", "Position across the three risk bands"),
        xaxis=dict(range=[0, 1], tickvals=[0, LOW_TH, HIGH_TH, 1], tickformat=".0%", title="Dropout probability", automargin=True),
        yaxis=dict(range=[-0.5, 1.0], showticklabels=False, showgrid=False, zeroline=False),
        height=250, margin=dict(l=20, r=20, t=76, b=56), showlegend=False, **_transparent(),
    )
    return fig


def plot_radar_profile(user_input, medians, top_features):
    labels, student_vals, cohort_vals = [], [], []
    for feat in top_features:
        val = float(user_input.get(feat, medians.get(feat, 0.0)))
        med = float(medians.get(feat, 0.0))
        denom = med if med > 0 else 1.0
        student_norm = max(0.0, min(1.0, val / denom)) if denom else 0.0
        labels.append(friendly_name(feat))
        student_vals.append(student_norm)
        cohort_vals.append(1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=student_vals + student_vals[:1], theta=labels + labels[:1], fill="toself", name="This student",
                                  line=dict(color=_C_NAVY, width=2.5), fillcolor="rgba(27, 42, 92, 0.22)"))
    fig.add_trace(go.Scatterpolar(r=cohort_vals + cohort_vals[:1], theta=labels + labels[:1], fill="toself", name="Cohort median",
                                  line=dict(color=_C_WARN, width=2.5), fillcolor="rgba(245, 165, 36, 0.14)"))
    fig.update_layout(
        title=_ttl("Student profile vs cohort median", "Each axis normalized to the cohort median (outer ring = at or above median)"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.5, 1], tickfont=dict(size=9), gridcolor="#E5EAF3"),
            angularaxis=dict(tickfont=dict(size=11.5, color=_C_INK), gridcolor="#E5EAF3"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=430, margin=dict(l=80, r=80, t=90, b=64), showlegend=True, legend=_legend_bottom(), paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _sweep_for(feature, medians):
    if feature in ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"):
        return np.arange(0, 21, 1)
    if "grade" in feature.lower():
        return np.arange(0, 20.5, 0.5)
    if feature in ("Age at enrollment",):
        return np.arange(17, 66, 1)
    if feature in ("Tuition fees up to date", "Scholarship holder", "Gender", "Debtor"):
        return np.array([0, 1])
    med = float(medians.get(feature, 0.0))
    return np.linspace(max(0, med - 5), med + 5, 25)


def plot_counterfactual_curve(base_input, feature, current_value, model, medians, feature_cols):
    sweep = _sweep_for(feature, medians)
    rows = []
    for v in sweep:
        row = dict(base_input)
        row[feature] = float(v)
        rows.append({f: row.get(f, float(medians[f])) for f in feature_cols})
    probs = np.asarray(model.predict_proba(pd.DataFrame(rows))[:, 1], dtype=float)
    cur_prob = float(model.predict_proba(pd.DataFrame([{f: base_input.get(f, float(medians[f])) for f in feature_cols}]))[:, 1][0])
    name = friendly_name(feature)
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=LOW_TH, fillcolor="rgba(18, 165, 148, 0.10)", line_width=0)
    fig.add_hrect(y0=LOW_TH, y1=HIGH_TH, fillcolor="rgba(245, 165, 36, 0.12)", line_width=0)
    fig.add_hrect(y0=HIGH_TH, y1=1.0, fillcolor="rgba(220, 63, 74, 0.10)", line_width=0)
    fig.add_trace(go.Scatter(x=sweep, y=probs, mode="lines+markers" if len(sweep) <= 2 else "lines",
                             line=dict(color=_C_BRAND, width=3), marker=dict(size=9, color=_C_BRAND), name="Predicted probability",
                             hovertemplate=f"{name}: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current_value], y=[cur_prob], mode="markers", marker=dict(size=14, color=_C_BAD, line=dict(width=2, color="white")),
                             name="Current value", hovertemplate=f"Current: {current_value}<br>Probability: {cur_prob:.1%}<extra></extra>"))
    xaxis = dict(title=name, automargin=True)
    if len(sweep) <= 2:
        xaxis.update(tickvals=[0, 1], range=[-0.15, 1.15])
    fig.update_layout(
        title=_ttl("Counterfactual sweep", f"{name}; shaded areas mark the Low / Medium / High risk bands"),
        xaxis=xaxis, yaxis=dict(title="Dropout probability", range=[0, 1], tickformat=".0%", automargin=True),
        height=380, margin=dict(l=20, r=20, t=80, b=84), showlegend=True, legend=_legend_bottom(), **_transparent(),
    )
    return fig


def plot_capacity_curve(probs_sorted):
    n = len(probs_sorted)
    k = np.arange(1, n + 1)
    cumulative = np.cumsum(probs_sorted)
    total = cumulative[-1]
    y = cumulative / total if total > 0 else np.zeros(n)
    cap = min(CAPACITY_LIMIT, n)
    y_cap = float(y[cap - 1]) if cap > 0 else 0.0
    rand = cap / n if n > 0 else 0.0
    flip = -1 if (n > 0 and cap / n > 0.7) else 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, n], y=[0, 1], mode="lines", line=dict(color="#ADB5BD", width=2, dash="dash"), name="Random selection", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=y, mode="lines", line=dict(color=_C_BRAND, width=3), name="Model-ranked selection",
                             hovertemplate="Intervened: %{x}<br>Risk captured: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cap, cap], y=[0, 1.06], mode="lines", line=dict(color=_C_BAD, width=2, dash="dot"),
                             name=f"Capacity C = {CAPACITY_LIMIT}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[cap], y=[y_cap], mode="markers", marker=dict(size=12, color=_C_BAD, line=dict(width=2, color="white")),
                             showlegend=False, hovertemplate=f"Top {cap} students<br>Risk captured: {y_cap:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cap], y=[rand], mode="markers", marker=dict(size=11, color="#6C757D", line=dict(width=2, color="white")),
                             showlegend=False, hovertemplate=f"Top {cap} by chance<br>Risk captured: {rand:.1%}<extra></extra>"))
    fig.add_annotation(x=cap, y=y_cap, ax=-70, ay=-30, xanchor="right", showarrow=True, arrowcolor=_C_BAD, arrowhead=2,
                       text=f"<b>Model: {y_cap:.1%}</b>", font=dict(size=12, color=_C_BAD), bgcolor="white", bordercolor=_C_BAD, borderwidth=1, borderpad=3)
    fig.add_annotation(x=cap, y=rand, ax=70 * flip, ay=30, xanchor="left" if flip > 0 else "right", showarrow=True, arrowcolor="#6C757D", arrowhead=2,
                       text=f"<b>Random: {rand:.1%}</b>", font=dict(size=12, color="#495057"), bgcolor="white", bordercolor="#6C757D", borderwidth=1, borderpad=3)
    fig.update_layout(
        title=_ttl("Capacity constraint", "Cumulative risk captured vs students intervened"),
        xaxis=dict(title="Students intervened (ranked by predicted risk)", automargin=True),
        yaxis=dict(title="Share of cohort risk captured", range=[0, 1.1], tickformat=".0%", automargin=True),
        height=450, margin=dict(l=20, r=30, t=80, b=88), showlegend=True, legend=_legend_bottom(), **_transparent(),
    )
    return fig


def plot_batch_band_distribution(bands):
    counts = pd.Series(bands).value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=[_band_hex(b) for b in counts.index],
        text=[f"{int(v):,}" for v in counts.values], textposition="outside", cliponaxis=False, textfont=dict(size=12, color=_C_INK),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=_ttl("Risk band distribution", "Students per band in the uploaded cohort"),
        xaxis=dict(title="", automargin=True),
        yaxis=dict(title="Number of students", range=[0, max(float(counts.max()), 1.0) * 1.22], automargin=True),
        height=350, margin=dict(l=20, r=20, t=76, b=44), showlegend=False, bargap=0.35, **_transparent(),
    )
    return fig


def plot_probability_histogram(probs):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs, xbins=dict(start=0, end=1, size=0.04), marker=dict(color=_C_BRAND, line=dict(color="white", width=1)),
                               opacity=0.9, hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>"))
    for xv, col, txt in [(LOW_TH, _C_OK, "Low | Med"), (HIGH_TH, _C_BAD, "Med | High")]:
        fig.add_vline(x=xv, line=dict(color=col, width=2, dash="dash"))
        fig.add_annotation(x=xv, y=1.0, yref="paper", yanchor="bottom", text=txt, showarrow=False, font=dict(size=11, color=col))
    fig.update_layout(
        title=_ttl("Distribution of predicted probabilities", "Dashed lines mark the band thresholds"),
        xaxis=dict(title="Dropout probability", tickformat=".0%", range=[0, 1], automargin=True),
        yaxis=dict(title="Number of students", automargin=True),
        height=350, margin=dict(l=20, r=20, t=96, b=48), showlegend=False, bargap=0.04, **_transparent(),
    )
    return fig


def plot_fairness_tpr():
    groups = ["Non-Scholarship", "Scholarship"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Unmitigated (global threshold)", x=groups, y=[0.907, 0.885], marker_color="#94A3B8",
                         text=["0.907", "0.885"], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(name="Mitigated (group thresholds)", x=groups, y=[0.849, 0.846], marker_color=_C_BRAND,
                         text=["0.849", "0.846"], textposition="outside", cliponaxis=False))
    fig.update_layout(
        title=_ttl("True positive rate by group", "Before and after group-specific thresholds"),
        barmode="group", yaxis=dict(range=[0, 1.12], title="TPR", automargin=True), xaxis=dict(automargin=True),
        height=360, margin=dict(l=20, r=20, t=80, b=64), showlegend=True, legend=_legend_bottom(), **_transparent(),
    )
    return fig


# ==========================================================
# PDF DESIGN SYSTEM
# ==========================================================
_PC = {
    "navy": colors.HexColor("#1B2A5C"),
    "brand": colors.HexColor("#3B5BDB"),
    "light": colors.HexColor("#EAF0FF"),
    "lighter": colors.HexColor("#F6F8FC"),
    "ink": colors.HexColor("#0E1B3D"),
    "text": colors.HexColor("#334155"),
    "muted": colors.HexColor("#64748B"),
    "line": colors.HexColor("#E2E8F0"),
    "ok": colors.HexColor("#12A594"),
    "warn": colors.HexColor("#F5A524"),
    "bad": colors.HexColor("#DC3F4A"),
}
_PAGE_W, _PAGE_H = A4
_MARGIN = 1.8 * cm
_CONTENT_W = _PAGE_W - 2 * _MARGIN


def _fig_to_png_bytes(fig, width=700, height=400):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _png_size(png):
    try:
        return struct.unpack(">II", png[16:24])
    except Exception:
        return (4, 3)


def _img(png, width_cm, max_h_cm=None):
    w, h = _png_size(png)
    width = width_cm * cm
    height = width * h / w
    if max_h_cm and height > max_h_cm * cm:
        height = max_h_cm * cm
        width = height * w / h
    im = RLImage(io.BytesIO(png), width=width, height=height)
    im.hAlign = "CENTER"
    return im


def _band_color(band):
    return {"Low Risk": _PC["ok"], "Medium Risk": _PC["warn"], "High Risk": _PC["bad"]}.get(band, _PC["navy"])


def _band_hex_str(band):
    return {"Low Risk": "#0B8A78", "Medium Risk": "#B7791F", "High Risk": "#C0303C"}.get(band, "#1B2A5C")


def _styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        kw.setdefault("fontName", "Helvetica")
        return ParagraphStyle(name, parent=base["BodyText"], **kw)

    return {
        "title": ps("PT", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.white),
        "subtitle": ps("PST", fontSize=11, leading=14, textColor=colors.HexColor("#C9D6FF")),
        "meta": ps("PM", fontSize=8.2, leading=11, textColor=colors.HexColor("#C9D6FF")),
        "h1": ps("PH1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=_PC["navy"], spaceBefore=14, spaceAfter=5),
        "h2": ps("PH2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=_PC["brand"], spaceBefore=8, spaceAfter=3),
        "body": ps("PB", fontSize=9.4, leading=14, textColor=_PC["text"], spaceAfter=4),
        "bullet": ps("PBL", fontSize=9.4, leading=13.5, textColor=_PC["text"], leftIndent=13, bulletIndent=2, spaceAfter=2),
        "small": ps("PS", fontSize=8, leading=11, textColor=_PC["muted"]),
        "caption": ps("PC", fontSize=8, leading=11, textColor=_PC["muted"], alignment=1, spaceAfter=6),
        "cell": ps("PCE", fontSize=8.3, leading=11, textColor=_PC["text"]),
        "cellh": ps("PCH", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white),
        "cellb": ps("PCB", fontName="Helvetica-Bold", fontSize=8.3, leading=11, textColor=_PC["ink"]),
        "kl": ps("PKL", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=_PC["muted"]),
        "kv": ps("PKV", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=_PC["ink"]),
        "callout": ps("PCO", fontSize=9.4, leading=14, textColor=_PC["ink"]),
    }


def _p(text, style):
    return Paragraph(text, style)


def _hr():
    return HRFlowable(width="100%", thickness=0.6, color=_PC["line"], spaceBefore=3, spaceAfter=6)


def _section(S, title):
    return KeepTogether([Paragraph(title, S["h1"]), HRFlowable(width="100%", thickness=1.2, color=_PC["brand"], spaceBefore=0, spaceAfter=6)])


def _bullets(S, items):
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def _banner(S, title, subtitle, meta):
    t = Table([[Paragraph(title, S["title"])], [Paragraph(subtitle, S["subtitle"])], [Paragraph(meta, S["meta"])]], colWidths=[_CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PC["navy"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16), ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (0, 0), 16), ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
        ("TOPPADDING", (0, 1), (-1, -1), 2), ("BOTTOMPADDING", (0, -1), (-1, -1), 14),
        ("LINEBELOW", (0, -1), (-1, -1), 3, _PC["brand"]),
    ]))
    return t


def _kpi_tiles(S, items, col_widths=None):
    n = len(items)
    widths = col_widths or [_CONTENT_W / n] * n
    labels = [Paragraph(str(lbl).upper(), S["kl"]) for lbl, _, _ in items]
    values = [Paragraph(str(val), S["kv"]) for _, val, _ in items]
    t = Table([labels, values], colWidths=widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), _PC["lighter"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i, (_, _, accent) in enumerate(items):
        style.append(("LINEABOVE", (i, 0), (i, 0), 3, accent))
        if i < n - 1:
            style.append(("LINEAFTER", (i, 0), (i, -1), 6, colors.white))
    t.setStyle(TableStyle(style))
    return t


def _cell(S, v, bold=False):
    if isinstance(v, str):
        return Paragraph(xml_escape(v), S["cellb"] if bold else S["cell"])
    return v


def _std_table(data, col_widths=None, header_bg=None, zebra=True, S=None):
    S = S or _styles()
    header_bg = header_bg or _PC["navy"]
    rows = [[Paragraph(xml_escape(str(h)), S["cellh"]) for h in data[0]]]
    for r in data[1:]:
        rows.append([_cell(S, c) for c in r])
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, _PC["line"]),
        ("BOX", (0, 0), (-1, -1), 0.4, _PC["line"]),
    ]
    if zebra:
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _PC["lighter"]]))
    t.setStyle(TableStyle(style))
    return t


def _kv_table(pairs, key_w=5.5 * cm, val_w=10.5 * cm):
    return _std_table([["Field", "Value"]] + [[str(k), str(v)] for k, v in pairs], col_widths=[key_w, val_w])


def _band_tag(S, band):
    return Paragraph(f'<font color="{_band_hex_str(band)}"><b>{xml_escape(band)}</b></font>', S["cell"])


def _callout(S, band, heading, body_html):
    t = Table([[Paragraph(f"<b>{heading}</b>", S["callout"])], [Paragraph(body_html, S["callout"])]], colWidths=[_CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PC["lighter"]),
        ("LINEBEFORE", (0, 0), (0, -1), 4, _band_color(band)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
    ]))
    return t


class _NumberedCanvas(rl_canvas.Canvas):
    doc_label = "Report"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self._draw_chrome(total)
            super().showPage()
        super().save()

    def _draw_chrome(self, total):
        page = self._pageNumber
        self.saveState()
        if page > 1:
            self.setStrokeColor(_PC["line"])
            self.setLineWidth(0.6)
            self.line(_MARGIN, _PAGE_H - 1.15 * cm, _PAGE_W - _MARGIN, _PAGE_H - 1.15 * cm)
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(_PC["navy"])
            self.drawString(_MARGIN, _PAGE_H - 0.95 * cm, "Capacity-Aware Decision Support System")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(_PC["muted"])
            self.drawRightString(_PAGE_W - _MARGIN, _PAGE_H - 0.95 * cm, self.doc_label)
        self.setStrokeColor(_PC["line"])
        self.line(_MARGIN, 1.25 * cm, _PAGE_W - _MARGIN, 1.25 * cm)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(_PC["muted"])
        self.drawString(_MARGIN, 0.8 * cm, "Decision support only. Human judgement remains essential.")
        self.drawRightString(_PAGE_W - _MARGIN, 0.8 * cm, f"Page {page} of {total}")
        self.restoreState()


def _canvas_for(label):
    return type("_Canvas_" + label.replace(" ", ""), (_NumberedCanvas,), {"doc_label": label})


def _new_doc(buffer, title):
    return SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=_MARGIN, rightMargin=_MARGIN, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=title, author="Capacity-Aware Decision Support System",
    )


def _relative_position(v, med):
    try:
        vf = float(v)
        if pd.isna(med) or med == 0:
            return "n/a"
        ratio = vf / med
        if ratio >= 1.5:
            return "Well above median"
        if ratio >= 1.1:
            return "Above median"
        if ratio >= 0.9:
            return "At median"
        if ratio >= 0.5:
            return "Below median"
        return "Well below median"
    except (TypeError, ValueError):
        return "n/a"


# ==========================================================
# PDF REPORT (SINGLE STUDENT)
# ==========================================================
def build_pdf_report(
    student_inputs,
    p_dropout,
    band,
    action,
    shap_vals,
    feature_cols,
    medians,
    model,
    counterfactual_feature=None,
    counterfactual_curve_png=None,
    radar_png=None,
    top_k=10,
    levers=None,
):
    buffer = io.BytesIO()
    doc = _new_doc(buffer, "Student Dropout Risk Report")
    S = _styles()
    story = []
    W = _CONTENT_W
    now = datetime.now()

    story.append(_banner(
        S, "Student Dropout Risk Report", "Capacity-Aware Decision Support System",
        f"Model: {xml_escape(MODEL_NAME)}  |  Version: {xml_escape(MODEL_VERSION)}  |  Generated: {now.strftime('%Y-%m-%d %H:%M')}",
    ))
    story.append(Spacer(1, 10))
    accent = _band_color(band)
    story.append(_kpi_tiles(S, [
        ("Dropout probability", f"{p_dropout:.1%}", accent),
        ("Risk band", f'<font color="{_band_hex_str(band)}">{xml_escape(band)}</font>', accent),
        ("Recommended action", xml_escape(action), accent),
    ], col_widths=[4.4 * cm, 4.2 * cm, W - 8.6 * cm]))

    top_pos = sorted([(feature_cols[i], shap_vals[i]) for i in range(len(shap_vals)) if shap_vals[i] > 0], key=lambda x: -x[1])[:3]
    top_neg = sorted([(feature_cols[i], shap_vals[i]) for i in range(len(shap_vals)) if shap_vals[i] < 0], key=lambda x: x[1])[:3]

    story.append(_section(S, "1. Executive summary"))
    story.append(_p(
        "This report presents a probabilistic dropout risk assessment for a single student, the factors driving that "
        "assessment, and the intervention the Decision Support System recommends under its capacity constraint. The "
        "prediction comes from an XGBoost classifier trained on 3,630 students with 36 features (cross-validated AUC above "
        f"0.950, Brier score 0.058). The model estimates a dropout probability of <b>{p_dropout:.1%}</b>, which places this "
        f"student in the <b>{xml_escape(band)}</b> tier.", S["body"]))
    story.append(_p("Key findings", S["h2"]))
    findings = []
    if top_pos:
        findings.append("<b>Strongest risk-increasing factors:</b> " + ", ".join(f"{xml_escape(friendly_name(f))} ({v:+.2f})" for f, v in top_pos) + ".")
    if top_neg:
        findings.append("<b>Strongest protective factors:</b> " + ", ".join(f"{xml_escape(friendly_name(f))} ({v:+.2f})" for f, v in top_neg) + ".")
    findings.append(f"<b>Recommended action:</b> {xml_escape(action)}. Capacity constraint: C = {CAPACITY_LIMIT} intensive mentoring slots university-wide.")
    story.extend(_bullets(S, findings))

    story.append(_section(S, "2. Probability and band placement"))
    gauge_png = _fig_to_png_bytes(plot_gauge(p_dropout, band), width=620, height=340)
    band_rows = [["Band", "Probability", "Standard action"]]
    ref = [("Low Risk", f"< {LOW_TH:.2f}"), ("Medium Risk", f"{LOW_TH:.2f} to < {HIGH_TH:.2f}"), ("High Risk", f">= {HIGH_TH:.2f}")]
    for b, rng in ref:
        band_rows.append([_band_tag(S, b), rng, get_intervention(b)])
    band_tbl = _std_table(band_rows, col_widths=[2.1 * cm, 2.6 * cm, W - 7.6 * cm - 4.7 * cm], S=S)
    hl = [i for i, (b, _) in enumerate(ref, start=1) if b == band]
    if hl:
        band_tbl.setStyle(TableStyle([("BACKGROUND", (0, hl[0]), (-1, hl[0]), _PC["light"])]))
    if gauge_png:
        layout = Table([[_img(gauge_png, 7.4), band_tbl]], colWidths=[7.6 * cm, W - 7.6 * cm])
        layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(layout)
    else:
        story.append(band_tbl)
    story.append(Spacer(1, 6))
    band_png = _fig_to_png_bytes(plot_threshold_explorer(p_dropout), width=800, height=260)
    if band_png:
        story.append(_img(band_png, 16.6))
        story.append(_p("Position of the student's predicted probability across the three risk bands.", S["caption"]))

    story.append(CondPageBreak(9 * cm))
    story.append(_section(S, "3. Submitted student profile"))
    story.append(_p("The values below were entered by the operator. All other model features (of 36 in total) were imputed from the training cohort median.", S["body"]))
    profile_rows = [["Indicator", "Submitted value", "Cohort median", "Relative position"]]
    for k, v in student_inputs.items():
        med = float(medians.get(k, np.nan))
        profile_rows.append([friendly_name(k), str(v), f"{med:.2f}" if not pd.isna(med) else "n/a", _relative_position(v, med)])
    story.append(_std_table(profile_rows, col_widths=[6.2 * cm, 3.4 * cm, 3.4 * cm, W - 13 * cm], S=S))
    if radar_png:
        story.append(Spacer(1, 8))
        story.append(_img(radar_png, 11.5, max_h_cm=8.6))
        story.append(_p("Profile versus cohort median. The outer ring marks a value at or above the median.", S["caption"]))

    story.append(CondPageBreak(12 * cm))
    story.append(_section(S, "4. Factor attribution (SHAP)"))
    story.append(_p(
        "SHAP values quantify the marginal contribution of each feature to the predicted probability. Positive values push the "
        "prediction toward dropout; negative values pull it away. Values are in log-odds space, so read them for direction and "
        "magnitude rather than as direct probability percentages.", S["body"]))
    shap_png = _fig_to_png_bytes(plot_shap_waterfall(feature_cols, shap_vals, top_n=top_k), width=820, height=520)
    if shap_png:
        story.append(_img(shap_png, 16.2, max_h_cm=10.2))
    story.append(_p("Top contributors", S["h2"]))
    idx = np.argsort(np.abs(shap_vals))[-top_k:][::-1]
    shap_rows = [["Rank", "Feature", "SHAP impact", "Direction", "Submitted", "Cohort median"]]
    for rank, i in enumerate(idx, start=1):
        v = float(shap_vals[i])
        feat = feature_cols[i]
        med_val = float(medians.get(feat, np.nan))
        direction = f'<font color="{"#C0303C" if v > 0 else "#0B8A78"}"><b>{"Increases risk" if v > 0 else "Reduces risk"}</b></font>'
        shap_rows.append([str(rank), friendly_name(feat), f"{v:+.3f}", Paragraph(direction, S["cell"]),
                          str(student_inputs.get(feat, "(imputed)")), f"{med_val:.2f}" if not pd.isna(med_val) else "n/a"])
    story.append(_std_table(shap_rows, col_widths=[1.2 * cm, 5.2 * cm, 2.3 * cm, 2.9 * cm, 2.6 * cm, W - 14.2 * cm], S=S))
    donut_png = _fig_to_png_bytes(plot_contribution_donut(shap_vals), width=560, height=400)
    if donut_png:
        story.append(Spacer(1, 8))
        story.append(_img(donut_png, 8.2))

    if levers:
        good = [l for l in levers if l["delta"] < -0.0005]
        story.append(CondPageBreak(7 * cm))
        story.append(_section(S, "5. Highest-impact levers"))
        story.append(_p("Modest single changes, each evaluated with all other indicators held constant, ranked by predicted reduction in dropout probability. "
                        "These are model associations, not causal effects.", S["body"]))
        if good:
            rows = [["Rank", "Change", "Indicator (from to)", "Predicted probability", "Change"]]
            for r, l in enumerate(good, start=1):
                rows.append([str(r), l["label"], f'{l["current"]} to {l["target"]}', f'{p_dropout:.1%} to {l["prob"]:.1%}',
                             Paragraph(f'<font color="#0B8A78"><b>{l["delta"] * 100:+.1f} pp</b></font>', S["cell"])])
            story.append(_std_table(rows, col_widths=[1.2 * cm, 6.2 * cm, 3.4 * cm, 4 * cm, W - 14.8 * cm], S=S))
        else:
            story.append(_p("No single modest change is predicted to reduce this student's risk.", S["body"]))

    if counterfactual_curve_png:
        story.append(CondPageBreak(10 * cm))
        story.append(_section(S, "6. Counterfactual analysis"))
        story.append(_p(
            "The curve shows how the predicted probability responds when a single indicator moves across its full range while all other "
            "features stay constant. Advisors can use it to see which changes would reduce risk the most.", S["body"]))
        if counterfactual_feature:
            story.append(_p(f"Perturbed feature: <b>{xml_escape(friendly_name(counterfactual_feature))}</b>", S["body"]))
        story.append(_img(counterfactual_curve_png, 16.2, max_h_cm=9.4))

    story.append(CondPageBreak(7 * cm))
    story.append(_section(S, "7. Recommended intervention"))
    story.append(_callout(S, band, f"Primary action: {xml_escape(action)}", xml_escape(get_intervention_detail(band))))
    story.append(Spacer(1, 6))
    story.append(_p("<b>Escalation rule.</b> If at the next monitoring checkpoint the student's predicted probability rises above the next band "
                    "threshold, escalate to the more intensive tier, provided capacity is available.", S["body"]))

    story.append(CondPageBreak(9 * cm))
    story.append(_section(S, "8. Methodology and limitations"))
    story.append(_p("Model", S["h2"]))
    story.append(_p("XGBoost gradient-boosted classifier with 400 trees, max depth 4, learning rate 0.05, subsample and colsample of 0.8, "
                    "and positive-class reweighting to address class imbalance.", S["body"]))
    story.append(_p("Data", S["h2"]))
    story.append(_p("Trained on the UCI Student Dropout and Academic Success dataset after removing the ambiguous 'Enrolled' class, yielding "
                    "3,630 students. An 80/20 stratified split was used, matching the paper's test cohort of roughly 726 students.", S["body"]))
    story.append(_p("Fairness", S["h2"]))
    story.append(_p("The system applies active Equal Opportunity mitigation by optimizing group-specific probability thresholds. In the "
                    "scholarship versus non-scholarship audit, the TPR gap narrowed from 0.022 (unmitigated) to 0.003 (mitigated).", S["body"]))
    story.append(_p("Limitations", S["h2"]))
    story.extend(_bullets(S, [
        "Predictions are probabilistic and reflect patterns in historical data; they are not causal claims about any individual student.",
        "Counterfactual and lever analyses assume one feature changes in isolation; real-world changes often co-occur.",
        f"Capacity C = {CAPACITY_LIMIT} is a prototype parameter and should be calibrated to the institution's actual advising resources.",
        "The system is a decision support tool and must not be used as an automated decision-maker; human judgement remains essential.",
    ]))
    story.append(Spacer(1, 12))
    story.append(_hr())
    story.append(_p(f"Generated automatically by a prototype Decision Support System. Report ID: {now.strftime('%Y%m%d%H%M%S')}", S["small"]))

    doc.build(story, canvasmaker=_canvas_for("Student Risk Report"))
    buffer.seek(0)
    return buffer


# ==========================================================
# PDF REPORT (BATCH)
# ==========================================================
def build_batch_pdf_report(
    scored_df,
    summary_df,
    curve_png,
    dist_png,
    hist_png,
    model_name=MODEL_NAME,
    top_n=50,
):
    buffer = io.BytesIO()
    doc = _new_doc(buffer, "Batch Cohort Risk Report")
    S = _styles()
    story = []
    W = _CONTENT_W
    now = datetime.now()
    total = max(len(scored_df), 1)

    n_high = int((scored_df["band"] == "High Risk").sum())
    n_med = int((scored_df["band"] == "Medium Risk").sum())
    n_low = int((scored_df["band"] == "Low Risk").sum())
    n_intensive = int((scored_df["action"] == "Intensive Mentoring and Counseling").sum())
    n_overflow = int((scored_df["action"].str.contains("capacity overflow", na=False)).sum())
    n_workshop = int((scored_df["action"] == "Skills Workshops and Progress Monitoring").sum())
    n_general = int((scored_df["action"] == "General Academic Support").sum())

    story.append(_banner(
        S, "Batch Cohort Risk Report", "Capacity-Aware Decision Support System",
        f"Model: {xml_escape(str(model_name))}  |  Version: {xml_escape(MODEL_VERSION)}  |  Generated: {now.strftime('%Y-%m-%d %H:%M')}  |  Cohort size: {len(scored_df):,}",
    ))
    story.append(Spacer(1, 10))
    sm = {str(k): str(v) for k, v in summary_df.items()}
    story.append(_kpi_tiles(S, [
        ("Students scored", f"{len(scored_df):,}", _PC["brand"]),
        ("High risk", sm.get("High risk", str(n_high)), _PC["bad"]),
        ("Medium risk", sm.get("Medium risk", str(n_med)), _PC["warn"]),
        ("Low risk", sm.get("Low risk", str(n_low)), _PC["ok"]),
    ]))
    story.append(Spacer(1, 6))
    story.append(_kpi_tiles(S, [
        ("Mean probability", sm.get("Mean probability", "n/a"), _PC["brand"]),
        ("Intensive slots used", sm.get("Intensive slots used", f"{n_intensive} / {CAPACITY_LIMIT}"), _PC["brand"]),
        ("Capacity overflow", sm.get("Capacity overflow", str(n_overflow)), _PC["warn"]),
        ("Risk captured at capacity", sm.get("Risk captured at capacity", "n/a") + "  (" + sm.get("Lift vs random", "n/a") + ")", _PC["ok"]),
    ]))

    story.append(_section(S, "1. Executive summary"))
    story.append(_p(
        f"This report summarizes the dropout risk profile of an uploaded cohort of <b>{len(scored_df):,} students</b>. All students were "
        f"scored in a single pass by the XGBoost classifier. Risk bands use the thresholds Low &lt; {LOW_TH:.2f}, Medium &lt; {HIGH_TH:.2f} "
        f"and High &gt;= {HIGH_TH:.2f}. The capacity constraint of C = {CAPACITY_LIMIT} intensive mentoring slots was applied to the "
        "highest-risk students, with overflow routed to scalable workshops.", S["body"]))
    story.append(_std_table([["Metric", "Value"]] + [[str(k), str(v)] for k, v in summary_df.items()], col_widths=[W * 0.55, W * 0.45], S=S))

    story.append(CondPageBreak(9 * cm))
    story.append(_section(S, "2. Cohort risk distribution"))
    imgs = [_img(p, 8.3) for p in (dist_png, hist_png) if p]
    if len(imgs) == 2:
        t = Table([imgs], colWidths=[W / 2, W / 2])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(t)
    else:
        story.extend(imgs)
    story.append(_p("Left: students per risk band. Right: distribution of predicted probabilities with band thresholds.", S["caption"]))

    story.append(CondPageBreak(12 * cm))
    story.append(_section(S, "3. Capacity constraint"))
    story.append(_p(
        "The curve plots the cumulative share of total cohort risk captured as more students are intervened on, ranked by predicted "
        "probability. A steep initial climb means the model concentrates risk in a small number of students, which is what a "
        f"capacity-limited system needs. The dotted vertical line marks the operational capacity C = {CAPACITY_LIMIT}; the grey dashed "
        "diagonal shows what a random selection of the same size would capture.", S["body"]))
    if curve_png:
        story.append(_img(curve_png, 16.4, max_h_cm=10.6))

    story.append(CondPageBreak(7 * cm))
    story.append(_section(S, "4. Capacity allocation breakdown"))
    alloc = [
        ["Tier / action", "Students", "Share of cohort", "Resource notes"],
        ["High risk (all)", f"{n_high:,}", f"{n_high / total:.1%}", "Top-priority tier"],
        ["   Intensive mentoring", f"{n_intensive:,}", f"{n_intensive / total:.1%}", f"Capped at C = {CAPACITY_LIMIT}"],
        ["   Overflow to workshops", f"{n_overflow:,}", f"{n_overflow / total:.1%}", "Capacity-aware reallocation"],
        ["Medium risk", f"{n_med:,}", f"{n_med / total:.1%}", "Workshops and monthly monitoring"],
        ["Low risk", f"{n_low:,}", f"{n_low / total:.1%}", "General academic support"],
    ]
    story.append(_std_table(alloc, col_widths=[5 * cm, 2.6 * cm, 3.2 * cm, W - 10.8 * cm], S=S))
    story.append(_p(f"Actions assigned: {n_intensive:,} intensive, {n_workshop:,} workshops (medium risk), {n_overflow:,} workshops (overflow), {n_general:,} general support.", S["small"]))

    story.append(PageBreak())
    story.append(_section(S, f"5. Priority queue (top {top_n} highest-risk students)"))
    story.append(_p("Ranked by predicted dropout probability. The student index matches the row order of the annotated CSV after sorting by probability.", S["body"]))
    rows = [["Rank", "Student idx", "Probability", "Band", "Recommended action"]]
    for rank, (_, row) in enumerate(scored_df.head(top_n).iterrows(), start=1):
        rows.append([str(rank), str(row.get("student_index", "")), f"{row['probability']:.2%}", _band_tag(S, str(row["band"])), str(row["action"])])
    story.append(_std_table(rows, col_widths=[1.3 * cm, 2.3 * cm, 2.4 * cm, 2.7 * cm, W - 8.7 * cm], S=S))

    story.append(PageBreak())
    story.append(_section(S, "6. Band membership detail"))
    for band_name in ["High Risk", "Medium Risk", "Low Risk"]:
        subset = scored_df[scored_df["band"] == band_name]
        story.append(_p(f'<font color="{_band_hex_str(band_name)}">{band_name}</font> ({len(subset):,} students)', S["h2"]))
        if len(subset) == 0:
            story.append(_p("No students in this band.", S["small"]))
            continue
        rws = [["Student idx", "Probability", "Action"]]
        for _, row in subset.head(60).iterrows():
            rws.append([str(row.get("student_index", "")), f"{row['probability']:.2%}", str(row["action"])])
        story.append(_std_table(rws, col_widths=[2.6 * cm, 2.6 * cm, W - 5.2 * cm], S=S))
        if len(subset) > 60:
            story.append(_p(f"Showing the first 60 of {len(subset):,} students. The full list is in the annotated CSV.", S["small"]))
        story.append(Spacer(1, 6))

    story.append(CondPageBreak(9 * cm))
    story.append(_section(S, "7. Methodology and limitations"))
    story.append(_p("Model", S["h2"]))
    story.append(_p("XGBoost gradient-boosted classifier trained on 3,630 students with 36 features. Cross-validated AUC above 0.950 and Brier score 0.058.", S["body"]))
    story.append(_p("Capacity constraint", S["h2"]))
    story.append(_p(f"The system solves an allocation problem that maximizes total predicted risk captured, subject to a hard capacity of C = {CAPACITY_LIMIT} "
                    "intensive mentoring slots. High-risk students beyond capacity are routed to scalable workshops so that no high-risk student is left without an action.", S["body"]))
    story.append(_p("Fairness", S["h2"]))
    story.append(_p("Active Equal Opportunity mitigation is applied at the threshold level. Across scholarship and non-scholarship groups, the TPR gap is reduced from 0.022 to 0.003.", S["body"]))
    story.append(_p("Limitations", S["h2"]))
    story.extend(_bullets(S, [
        "Predictions reflect historical patterns and are not causal claims.",
        "Batch scoring assumes uploaded rows match the training feature schema; missing columns are imputed from cohort medians.",
        "The capacity parameter is a prototype value and should be tuned to institutional resources.",
        "Decisions must remain human-supervised.",
    ]))
    story.append(Spacer(1, 12))
    story.append(_hr())
    story.append(_p(f"Generated automatically by a prototype Decision Support System. Report ID: {now.strftime('%Y%m%d%H%M%S')}", S["small"]))

    doc.build(story, canvasmaker=_canvas_for("Cohort Risk Report"))
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
# HIGHEST-IMPACT LEVERS (new feature)
# ==========================================================
def compute_top_levers(user_input, model, medians, feature_cols, base_prob):
    specs = [
        ("Curricular units 1st sem (approved)", "+2 passed units (1st semester)", lambda v: min(20, v + 2)),
        ("Curricular units 2nd sem (approved)", "+2 passed units (2nd semester)", lambda v: min(20, v + 2)),
        ("Curricular units 2nd sem (grade)", "+2 points average grade (2nd semester)", lambda v: min(20.0, v + 2.0)),
        ("Tuition fees up to date", "Bring tuition fees up to date", lambda v: 1),
        ("Scholarship holder", "Secure a scholarship", lambda v: 1),
    ]
    rows, meta = [], []
    for feat, label, fn in specs:
        if feat not in feature_cols:
            continue
        cur = float(user_input.get(feat, medians[feat]))
        new = float(fn(cur))
        if new == cur:
            continue
        row = dict(user_input)
        row[feat] = new
        rows.append({f: row.get(f, float(medians[f])) for f in feature_cols})
        fmt = (lambda x: "Yes" if x == 1 else "No") if feat in ("Scholarship holder",) else (
            (lambda x: "Up to date" if x == 1 else "Overdue") if feat == "Tuition fees up to date" else (
                (lambda x: f"{x:.1f}") if "grade" in feat else (lambda x: f"{x:.0f}")))
        meta.append((label, fmt(cur), fmt(new)))
    if not rows:
        return []
    probs = np.asarray(model.predict_proba(pd.DataFrame(rows))[:, 1], dtype=float)
    out = [
        {"label": m[0], "current": m[1], "target": m[2], "prob": float(p), "delta": float(p - base_prob)}
        for m, p in zip(meta, probs)
    ]
    return sorted(out, key=lambda d: d["delta"])


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
              <div class="hero-sub">Interpretable Student Retention Prototype</div>
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
            <div class="side-t">Risk bands</div>
            <div class="lg"><div><i style="background:#12A594"></i>Low risk</div><span>&lt; {LOW_TH:.0%}</span></div>
            <div class="lg"><div><i style="background:#F5A524"></i>Medium risk</div><span>{LOW_TH:.0%} to {HIGH_TH:.0%}</span></div>
            <div class="lg"><div><i style="background:#DC3F4A"></i>High risk</div><span>&ge; {HIGH_TH:.0%}</span></div>
            <div class="side-t">Capacity</div>
            <div class="side-box"><b>{CAPACITY_LIMIT}</b> intensive mentoring slots university-wide. High-risk students beyond the cap are routed to workshops and monitoring.</div>
            <div class="side-t">Model</div>
            <div class="side-box"><b>{MODEL_NAME}</b><br>{MODEL_VERSION}</div>
            <div class="side-t">Use with care</div>
            <div class="side-box">Decision support only. Predictions are probabilistic and human judgement remains essential.</div>
            """
        )


def render_levers(levers, p_dropout):
    good = [l for l in (levers or []) if l["delta"] < -0.0005]
    _title("Highest-impact levers", "Modest single changes, ranked by the predicted reduction in dropout probability.")
    if not good:
        _hint("No single modest change is predicted to reduce this student's risk. Try combinations in the simulator below.")
        return
    top = max(-g["delta"] for g in good)
    rows = ""
    for i, g in enumerate(good, 1):
        pct = max(8, int(round(100 * (-g["delta"]) / top)))
        rows += (
            f'<div class="lev-r"><div class="lev-n">{i}</div>'
            f'<div style="min-width:0"><div class="lev-t">{g["label"]}</div>'
            f'<div class="lev-d">{g["current"]} to {g["target"]} &nbsp;|&nbsp; {p_dropout:.1%} to {g["prob"]:.1%}</div></div>'
            f'<div class="lev-v"><b>{g["delta"] * 100:+.1f} pp</b><div class="lev-track"><div class="lev-fill" style="width:{pct}%"></div></div></div></div>'
        )
    _md('<div class="lev">' + rows + "</div>")
    _hint("Estimates come from model associations, not causal effects. Use them to prioritise the conversation with the student.")


def render_history():
    hist = st.session_state.get("history", [])
    with st.expander(f"Assessment history (this session)  |  {len(hist)} saved", expanded=False):
        if not hist:
            st.caption("Assessments you run are listed here so you can compare students. History is cleared when the session ends.")
            return
        hdf = pd.DataFrame(hist)
        kwargs = {}
        if hasattr(st, "column_config"):
            kwargs["hide_index"] = True
        st.dataframe(hdf, use_container_width=True, **kwargs)
        h1, h2 = st.columns(2)
        with h1:
            st.download_button("Download History CSV", data=hdf.to_csv(index=False).encode("utf-8"),
                               file_name="assessment_history.csv", mime="text/csv", use_container_width=True, key="dl_history")
        with h2:
            if st.button("Clear history", use_container_width=True, key="clear_history"):
                st.session_state["history"] = []
                (getattr(st, "rerun", None) or st.experimental_rerun)()


# ==========================================================
# RESULT RENDERERS
# ==========================================================
def render_empty_state():
    _md(
        f"""
        <div class="empty">
          <div class="empty-i">{EMPTY_SVG}</div>
          <div class="empty-t">No assessment yet</div>
          <div class="empty-s">Enter the student's metrics and click 'Generate Diagnostic Prediction' to view
          capacity-aware interventions, SHAP attributions, counterfactual analysis, and to download a detailed PDF report.</div>
          <div class="empty-c"><span>Risk band</span><span>SHAP drivers</span><span>Cohort comparison</span><span>Top levers</span><span>What-if simulator</span><span>PDF report</span></div>
        </div>
        """
    )


def render_assessment(res, model, medians, feature_cols):
    p_dropout = res["p"]
    band = res["band"]
    action = res["action"]
    shap_vals = res["shap_vals"]
    user_input = res["user_input"]
    radar_fig = res["radar_fig"]
    levers = res.get("levers")
    tone = TONE.get(band, "bad")

    c_res, c_dl = st.columns([3, 1.2], gap="medium")
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
        st.download_button(
            label="Download PDF Report",
            data=res["pdf"],
            file_name=res["pdf_name"],
            mime="application/pdf",
            use_container_width=True,
            key="dl_single_pdf",
        )
        st.caption(f"Report generated {res['stamp']}")

    t_over, t_drv, t_cmp, t_wif, t_act = st.tabs(
        ["Overview", "Risk drivers", "Cohort comparison", "What-if", "Action plan"]
    )

    with t_over:
        _hint(
            "The gauge shows the model's estimated probability that this student will drop out. "
            "The colored background bands correspond to the Low, Medium, and High risk thresholds used by the DSS."
        )
        col_gauge, col_band = st.columns([1, 1.4], gap="medium")
        with col_gauge:
            st.plotly_chart(plot_gauge(p_dropout, band), use_container_width=True, config={"displayModeBar": False})
        with col_band:
            st.plotly_chart(plot_threshold_explorer(p_dropout), use_container_width=True, config={"displayModeBar": False})

    with t_drv:
        _hint(
            'Each bar shows how much a feature pushed the prediction <b style="color:#DC3F4A;">up (risk)</b> or '
            '<b style="color:#12A594;">down (protection)</b>. Hover a bar to see the exact impact value.'
        )
        col_water, col_donut = st.columns([1.6, 1], gap="medium")
        with col_water:
            st.plotly_chart(plot_shap_waterfall(feature_cols, shap_vals, top_n=6), use_container_width=True, config={"displayModeBar": False})
        with col_donut:
            st.plotly_chart(plot_contribution_donut(shap_vals), use_container_width=True, config={"displayModeBar": False})

        top_pos = [(friendly_name(feature_cols[i]), shap_vals[i]) for i in np.argsort(shap_vals)[::-1] if shap_vals[i] > 0][:3]
        top_neg = [(friendly_name(feature_cols[i]), shap_vals[i]) for i in np.argsort(shap_vals) if shap_vals[i] < 0][:3]
        chips_html = ""
        for name, val in top_pos:
            chips_html += f'<span class="factor-chip chip-risk">{name} ({val:+.2f})</span>'
        for name, val in top_neg:
            chips_html += f'<span class="factor-chip chip-prot">{name} ({val:+.2f})</span>'
        _group("Quick read")
        _md(chips_html if chips_html else "<i>No dominant factors.</i>")

    with t_cmp:
        _hint(
            "The dark shape is the student, normalized against the cohort median (yellow ring). "
            "Values reaching the outer ring mean the student is at or above the median for that indicator."
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    with t_wif:
        render_levers(levers, p_dropout)
        _group("What-if simulator")
        _hint("Move a lever below to simulate a change in the student's profile and immediately see how the predicted risk responds.")
        actionable_features = [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Tuition fees up to date",
            "Scholarship holder",
            "Age at enrollment",
        ]
        actionable_features = [f for f in actionable_features if f in feature_cols]
        cA, cB = st.columns([1, 1.3], gap="medium")
        with cA:
            cf_feature = st.selectbox("Select a feature to perturb", options=actionable_features, format_func=friendly_name, key="cf_feature")
        with cB:
            if cf_feature in ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"):
                cf_value = st.slider(f"Simulated value for {friendly_name(cf_feature)}", 0, 20, int(user_input.get(cf_feature, 0)), 1, key="cf_value_int")
            elif "grade" in cf_feature.lower():
                cf_value = st.slider(f"Simulated value for {friendly_name(cf_feature)}", 0.0, 20.0, float(user_input.get(cf_feature, 10.0)), 0.5, key="cf_value_float")
            elif cf_feature == "Age at enrollment":
                cf_value = st.slider(f"Simulated value for {friendly_name(cf_feature)}", 17, 65, int(user_input.get(cf_feature, 20)), 1, key="cf_value_age")
            else:
                cf_value = st.selectbox(
                    f"Simulated value for {friendly_name(cf_feature)}",
                    options=[("No", 0), ("Yes", 1)] if cf_feature != "Scholarship holder" else [("Yes", 1), ("No", 0)],
                    format_func=lambda x: x[0], key="cf_value_bin",
                )[1]

        cf_prob = compute_counterfactual(user_input, cf_feature, cf_value, model, medians, feature_cols)
        delta, delta_cls, arrow, band_note = band_change_summary(p_dropout, cf_prob)
        sim_tone = "ok" if delta < -0.005 else "bad" if delta > 0.005 else ""
        _kpi_grid([
            _kpi("Baseline probability", f"{p_dropout:.1%}", assign_risk_band(p_dropout)),
            _kpi("Simulated probability", f"{cf_prob:.1%}", f'<span class="{delta_cls}">{delta:+.2%}</span> {arrow}', sim_tone),
            _kpi("Risk band shift", assign_risk_band(cf_prob), band_note.strip() or "No band change", TONE.get(assign_risk_band(cf_prob), "")),
        ])
        _hint(
            f"Moving <b>{friendly_name(cf_feature)}</b> from <b>{user_input.get(cf_feature, 'current')}</b> to <b>{cf_value}</b> "
            f'would change the predicted dropout probability by <span class="{delta_cls}">{delta:+.2%}</span> {arrow}.{band_note}'
        )
        current_val = user_input.get(cf_feature, cf_value)
        st.plotly_chart(
            plot_counterfactual_curve(user_input, cf_feature, current_val, model, medians, feature_cols),
            use_container_width=True, config={"displayModeBar": False},
        )

    with t_act:
        _md(
            f"""
            <div class="result tone-{tone}">
              <div class="result-r" style="padding:1.1rem 1.4rem;">
                <span class="pill tone-{tone}">{band}</span>
                <div class="result-ak">Recommended action</div>
                <div class="result-a">{action}</div>
                <div class="fair-d" style="margin:.4rem 0 0;font-size:.9rem;color:#4A5876;">{get_intervention_detail(band)}</div>
              </div>
            </div>
            """
        )
        _hint(
            "The counterfactual explorer and the highest-impact levers can identify one or two indicators to discuss with the student. "
            "Target those indicators first before escalating to intensive support."
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

    curve_png = _fig_to_png_bytes(cap_fig, width=850, height=520)
    dist_png = _fig_to_png_bytes(dist_fig, width=750, height=440)
    hist_png = _fig_to_png_bytes(hist_fig, width=750, height=440)

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
        st.download_button("Download Detailed Batch PDF Report", data=b["pdf"], file_name=b["pdf_name"],
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
        col_input, col_results = st.columns([1.1, 2], gap="large")

        with col_input:
            with _card("card-input"):
                _title("Student profile", "Enter the indicators, then run the assessment.")
                user_input = {}

                _group("Academic indicators")
                user_input["Curricular units 1st sem (approved)"] = st.slider(
                    "1st Semester Passed Units", 0, 20,
                    int(X_train["Curricular units 1st sem (approved)"].median()),
                    help="Number of curricular units the student passed in semester 1.",
                )
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
                    radar_png = _fig_to_png_bytes(radar_fig, width=760, height=560)

                    cf_feature_default = "Curricular units 2nd sem (approved)"
                    cf_value_default = user_input.get(cf_feature_default, 0)
                    cf_fig = plot_counterfactual_curve(
                        user_input, cf_feature_default, cf_value_default, model, medians, feature_cols
                    )
                    cf_png = _fig_to_png_bytes(cf_fig, width=900, height=470)

                    levers = compute_top_levers(user_input, model, medians, feature_cols, p_dropout)

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
                        levers=levers,
                    )
                    now = datetime.now()
                    st.session_state["assessment"] = {
                        "user_input": dict(user_input),
                        "p": p_dropout,
                        "band": band,
                        "action": action,
                        "shap_vals": shap_vals,
                        "radar_fig": radar_fig,
                        "levers": levers,
                        "pdf": pdf_buffer.getvalue(),
                        "pdf_name": f"dropout_risk_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf",
                        "stamp": now.strftime("%H:%M:%S"),
                    }
                    history = st.session_state.get("history", [])
                    history.append({
                        "Time": now.strftime("%H:%M:%S"),
                        "Probability": f"{p_dropout:.1%}",
                        "Band": band,
                        "Action": action,
                        "1st sem units": user_input["Curricular units 1st sem (approved)"],
                        "2nd sem units": user_input["Curricular units 2nd sem (approved)"],
                        "2nd sem grade": user_input["Curricular units 2nd sem (grade)"],
                        "Tuition up to date": "Yes" if user_input["Tuition fees up to date"] == 1 else "No",
                        "Scholarship": "Yes" if user_input["Scholarship holder"] == 1 else "No",
                        "Age": user_input["Age at enrollment"],
                    })
                    st.session_state["history"] = history

            res = st.session_state.get("assessment")
            if res is None:
                render_empty_state()
            else:
                render_assessment(res, model, medians, feature_cols)
            render_history()

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
                    '<div class="sec-s">One row per student. Columns must match the model\'s feature names. '
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
                run_clicked = st.button("Run Batch Scoring", type="primary", use_container_width=True)

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
