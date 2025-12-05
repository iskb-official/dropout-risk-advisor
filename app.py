# app.py - CORRECTED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ========== PAGE CONFIG MUST BE FIRST ==========
st.set_page_config(
    page_title="Dropout Risk Advisor",
    page_icon="🎓",
    layout="wide"
)

# ========== REST OF THE APP ==========
st.title("🎓 Dropout Risk Advisor")
st.success("✅ TEST VERSION - App is loading...")

# Check Python version
st.write(f"**Python version:** {sys.version}")

# List files in directory
st.write("**Files in directory:**")
try:
    files = os.listdir('.')
    for file in files:
        size = os.path.getsize(file) if os.path.isfile(file) else "DIR"
        st.write(f"- {file} ({size})")
except Exception as e:
    st.write(f"Error listing files: {e}")

# Try to load CSV
csv_path = "students_dropout_academic_success.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.success(f"✅ CSV loaded successfully!")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        
        if 'target' in df.columns:
            st.write("**Target distribution:**")
            st.write(df['target'].value_counts())
        
        st.write("**First 3 rows:**")
        st.dataframe(df.head(3))
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.write("Error details:", str(e))
else:
    st.error(f"❌ CSV file not found at: {csv_path}")
    st.info("💡 Make sure 'students_dropout_academic_success.csv' is uploaded to GitHub repository")

st.balloons()
st.success("🚀 App is working! Ready to add ML features.")

# Test button
if st.button("Test Button - Click Me"):
    st.success("✅ Button works! Everything is functional.")
