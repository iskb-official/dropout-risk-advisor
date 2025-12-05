Dropout Risk Advisor
A trustworthy AI-based decision-support system that helps higher-education institutions identify at-risk students early and recommend targeted interventions with transparent, explainable logic.

✨ Key Features
🤖 XGBoost Dropout Prediction
Trained on a real Portuguese higher-education dataset (4,424 students, 36 features) to estimate individual dropout probability.

🎯 Three-Tier Risk Classification
Converts probabilities into low, medium, and high risk bands using calibrated thresholds:

Low: p < 0.20

Medium: 0.20 ≤ p < 0.50

High: p ≥ 0.50

💡 Actionable Intervention Recommendations
Each risk band is mapped to a concrete support strategy:

High → Intensive mentoring and academic counseling

Medium → Skills workshops and progress monitoring

Low → No targeted action beyond general support

🔍 SHAP-Based Explainability
For every prediction, the advisor highlights the top features that increase or reduce risk, using SHAP values and a plain-language explanation.

⚖️ Fairness-Aware Analysis
Includes a simple fairness audit over the test set, reporting error and intervention rates by gender and scholarship status, to support transparent institutional discussion.

🌐 Deployable Prototype UI
Built with Streamlit, including integrated help text (feature meanings, decision logic, fairness snapshot) suitable for demonstrations and small user studies.

📊 Model Overview
Algorithm: XGBoost (gradient boosted trees)
Task: Binary classification

1 = Dropout

0 = Non-dropout (Graduate + Enrolled)

Performance (held-out test set):

ROC–AUC ≈ 0.93

F1 (dropout class): ≈ 0.81

High-risk band (~33% of students) contains ≈ 80% actual dropouts

Low-risk band has ≈ 5% dropout rate

Key predictive drivers (top mean absolute SHAP values) include:

Curricular units (approved) in 1st and 2nd semesters

2nd-semester average grade

Tuition-fee status (up to date vs overdue)

Course, admission grade, age at enrollment, and economic indicators

🌐 Live Demo
You can try the advisor directly in your browser:

👉 Live demo: https://dropoutra.streamlit.app/

The demo lets you:

Enter a synthetic student profile (selected key features)

View the predicted dropout probability and risk band

See the recommended intervention

Inspect the top SHAP-based factors driving the prediction

Read the decision logic and fairness snapshot used in the study

🚀 Quick Start
Local Development
bash
# Clone repository
git clone https://github.com/iskb-official/dropout-risk-advisor.git
cd dropout-risk-advisor

# (Optional but recommended) create and activate a virtual environment
# python -m venv .venv
# source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
The app will open at http://localhost:8501 in your browser.

📁 Project Structure
text
dropout-risk-advisor/
├─ app.py                         # Streamlit prototype UI
├─ baseline_dropout.py            # Logistic regression baseline
├─ dropout_models_step2.py        # LR + XGBoost training and comparison
├─ dropout_risk_bands_step3.py    # Risk band derivation (low/medium/high)
├─ dropout_interventions_step4.py # Intervention policy + capacity constraint
├─ dropout_shap_step5.py          # SHAP explainability and rationales
├─ fairness_analysis_step6.py     # Fairness metrics by gender/scholarship
├─ students_dropout_academic_success.csv  # Portuguese HE dataset (UCI/Kaggle)
├─ xgb_feature_importances.csv    # XGBoost feature importance (gain)
├─ shap_global_importance.csv     # Global mean |SHAP| feature ranking
├─ risk_bands_test.csv            # Test-set probabilities and bands
├─ dropout_decisions_test.csv     # Test-set decisions and interventions
├─ dropout_decisions_with_shap_test.csv  # Decisions + SHAP rationales
└─ requirements.txt               # Python dependencies
(Some files may be generated after running the scripts.)

🧩 How the Advisor Works
1. Risk Prediction
An XGBoost model estimates P(dropout) from demographic, academic, and economic features.

2. Risk Banding
Probabilities are mapped to low/medium/high bands using fixed thresholds (0.20 and 0.50), calibrated on validation data.

3. Intervention Policy
Each band triggers a default action:

High → Intensive mentoring and counseling

Medium → Skills workshop + monitoring

Low → No targeted action

A capacity constraint can prioritize only the top‑N highest-risk students within the high band for the most intensive support.

4. Explanation Layer
SHAP TreeExplainer produces per-student feature attributions. The app displays the top 3 features (by |SHAP|) and whether each increases or reduces risk, forming a natural-language explanation.

5. Fairness Audit (Offline Analysis)
On the held-out test set, the tool computes:

TPR and FPR for the "high-risk" decision rule

Share of high-risk and intensive-mentoring assignments by gender and scholarship status

These metrics are included in the prototype's fairness snapshot and in the research write-up.

🧪 Intended Use
This project is designed as a research and demonstration tool for:

Educational data mining / learning analytics studies

Prototyping trustworthy AI decision-support systems

Exploring explainable AI and fairness concepts in higher education

⚠️ Important Note: It is not a production-ready system and should not be used as the sole basis for real student decisions without appropriate governance, calibration, and institutional review.

📜 Data and Licensing
Dataset: Predict Students' Dropout and Academic Success (Portuguese higher education), available from UCI / Kaggle under their respective terms.

Code: This repository is released under the MIT License (see LICENSE).

🤝 Contributing
Contributions and suggestions are welcome, especially on:

Additional datasets (e.g., MOOC/LMS logs)

Alternative models or banding strategies

Fairness constraints and evaluation

UI/UX improvements for advisors and administrators

Please open an issue or pull request to discuss proposed changes.

📬 Contact
For questions, collaboration, or citation details, please contact:

Md Shakib Hasan
Faculty of Artificial Intelligence in Education
Central China Normal University
