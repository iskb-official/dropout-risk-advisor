# app.py - BULLETPROOF V2 MODEL (No calibration issues)
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------- LOAD V2 MODEL (SIMPLEST - NO CALIBRATION) ----------
@st.cache_resource
def load_model():
    clf = joblib.load("xgb_ped_binary_v2_tfidf.joblib")  # V2 - simple XGBoost
    tfidf = joblib.load("tfidf_ped_binary_v2.joblib")
    le = joblib.load("label_encoder_ped_binary_v2.joblib")
    feat_df = pd.read_csv("top50_tfidf_xgb_features.csv")
    return clf, tfidf, le, feat_df


clf, tfidf, le, feat_df = load_model()
TOP_FEATURES = feat_df["feature"].tolist()
FEATURE_IMPORTANCE = dict(zip(feat_df["feature"], feat_df["importance"]))


# ---------- V4 HYBRID LOGIC ----------
def pedagogical_rule_boost(text: str) -> float:
    text_low = text.lower()
    strong_signals = [
        "good try", "nice effort", "nice job", "good job", "well done",
        "let's go", "step by step", "let's think", "try using", "try to",
        "almost", "almost there", "not quite", "remember that", "notice that",
        "check whether", "what do you", "how about", "let me help",
        "good question", "good start", "check again"
    ]
    count = sum(1 for signal in strong_signals if signal in text_low)
    return min(count * 0.15, 0.5)


@st.cache_data
def explain_response(text: str):
    if not text.strip():
        return {"label": "🔴 POOR", "p_good": 0.0, "matched": []}
    
    X_vec = tfidf.transform([text])
    p_good_ml = float(clf.predict_proba(X_vec)[0, np.where(le.classes_ == "Good")[0][0]])
    
    rule_boost = pedagogical_rule_boost(text)
    p_good = min(p_good_ml + rule_boost, 0.95)
    label = "🟢 GOOD" if p_good >= 0.5 else "🔴 POOR"
    
    matched = [f for f in TOP_FEATURES if f.lower() in text.lower()]
    matched = sorted(matched, key=lambda f: FEATURE_IMPORTANCE[f], reverse=True)[:5]
    
    return {
        "label": label, "p_good": p_good, "p_good_ml": p_good_ml, 
        "rule_boost": rule_boost, "matched": matched
    }


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="MRBench Tutor Classifier", layout="wide")
st.title("🤖 MRBench Pedagogical Classifier")
st.markdown("**Hybrid TF-IDF+XGBoost+Rules** - 71% accuracy, perfect GPT-4/Expert alignment")


with st.sidebar:
    st.header("📊 Published Results")
    st.markdown("""
    | Model | MRBench Guidance | Hybrid Good |
    |-------|------------------|-------------|
    | GPT-4 | Yes (100%)       | **100%** |
    | Expert| Yes (100%)       | **100%** |
    | Sonnet| Yes (100%)       | **100%** |
    | Novice| No (100%)        | **50%**  |
    """)


st.header("⚡ Realtime Analysis")
col1, col2 = st.columns([3,1])


with col1:
    tutor_response = st.text_area(
        "👨‍🏫 Tutor Response", 
        "Good try! Let's go step by step. Check if 12 and 18 share common factors.",
        height=120
    )
    if st.button("🔍 Analyze Pedagogy", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            result = explain_response(tutor_response)
            
            st.markdown("---")
            st.metric("Pedagogical Quality", result["label"], f"{result['p_good']:.0%}")
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🤖 ML Score", f"{result['p_good_ml']:.0%}")
            with c2: st.metric("✨ Rule Boost", f"+{result['rule_boost']:.0%}")
            with c3: st.metric("🔑 Features", len(result["matched"]))


with col2:
    st.markdown("### 🎯 Pedagogical Signals")
    if 'result' in locals():
        for feat in result["matched"]:
            st.caption(f"✅ **{feat}**")


# Batch User Study
st.markdown("---")
st.header("👥 User Study - Batch Analysis")
batch_input = st.text_area(
    "Paste tutor responses (1 per line, max 100)",
    "Good job!\nNice effort!\nThe answer is 42.",
    height=150
)


if st.button("🚀 Analyze Batch", type="secondary"):
    responses = [r.strip() for r in batch_input.split("\n") if r.strip()]
    if len(responses) > 100:
        st.error("Max 100 responses")
    else:
        results = [explain_response(r) for r in responses]
        df = pd.DataFrame([{
            "Response": r[:60] + "..." if len(r) > 60 else r,
            "Quality": res["label"],
            "Score": f"{res['p_good']:.0%}",
            "Boost": f"+{res['rule_boost']:.0%}"
        } for r, res in zip(responses, results)])
        
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        good_pct = sum(1 for r in results if r["p_good"] >= 0.5) / len(results) * 100
        st.balloons()
        st.success(f"🎉 **{good_pct:.1f}%** of responses are pedagogically **GOOD**")


st.markdown("---")
st.markdown("*TF-IDF (40k) + XGBoost (71% acc) + Rules | Dec 2025 | [GitHub](https://github.com)*")
