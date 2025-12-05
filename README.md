# Dropout Risk Advisor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dropoutra.streamlit.app/)

A trustworthy AI-based decision-support system that helps higher-education institutions identify at-risk students early and recommend targeted interventions with transparent, explainable logic.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🤖 XGBoost Prediction** | Trained on 4,424 students with 36 features to estimate dropout probability |
| **🎯 Three-Tier Risk Classification** | Low (<0.20), Medium (0.20-0.50), High (≥0.50) risk bands |
| **💡 Actionable Interventions** | Band-specific support strategies with capacity constraints |
| **🔍 SHAP Explainability** | Top 3 factors driving each prediction with plain-language explanations |
| **⚖️ Fairness Audit** | Error and intervention rates by gender and scholarship status |
| **🌐 Streamlit Prototype** | Interactive UI for demonstrations and user studies |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.93 |
| **F1 (Dropout Class)** | 0.81 |
| **High-Risk Band Coverage** | ~80% of actual dropouts |
| **Low-Risk Band Dropout Rate** | ~5% |

### Top Predictive Features
1. Curricular units approved (1st & 2nd semesters)
2. 2nd-semester average grade
3. Tuition fee status (up to date vs overdue)
4. Course, admission grade, and age at enrollment
5. Economic indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/iskb-official/dropout-risk-advisor.git
cd dropout-risk-advisor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

## 📁 Project Structure

```
dropout-risk-advisor/
├── app.py                          # Main Streamlit application
├── baseline_dropout.py             # Logistic regression baseline model
├── dropout_models_step2.py         # Model training and comparison
├── dropout_risk_bands_step3.py     # Risk band derivation
├── dropout_interventions_step4.py  # Intervention policy implementation
├── dropout_shap_step5.py           # SHAP explanations generation
├── fairness_analysis_step6.py      # Fairness metrics calculation
├── students_dropout_academic_success.csv  # Dataset (UCI/Kaggle)
├── xgb_feature_importances.csv     # XGBoost feature importance
├── shap_global_importance.csv      # Global SHAP importance
├── risk_bands_test.csv             # Test set predictions with risk bands
├── dropout_decisions_test.csv      # Intervention decisions
├── dropout_decisions_with_shap_test.csv  # Decisions with SHAP rationales
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🧠 How It Works

### 1. **Risk Prediction Pipeline**
```python
# Simplified prediction flow
student_features → XGBoost Model → Dropout Probability → Risk Band → Intervention
```

### 2. **Risk Classification**
- **Low Risk (p < 0.20)**: General institutional support only
- **Medium Risk (0.20 ≤ p < 0.50)**: Skills workshops + progress monitoring
- **High Risk (p ≥ 0.50)**: Intensive mentoring + academic counseling

### 3. **Explainability**
```python
# SHAP-based explanation example
Top factors for Student #123:
1. Low 2nd-semester grades (+25% risk)
2. Tuition fee overdue (+18% risk)  
3. High age at enrollment (+12% risk)
```

### 4. **Fairness Monitoring**
- **Gender**: Male vs Female intervention rates
- **Scholarship**: With vs Without scholarship error rates
- **Statistical parity**: High-risk assignment rates across groups

## 🌐 Live Demo

Try the interactive prototype: **[https://dropoutra.streamlit.app/](https://dropoutra.streamlit.app/)**

### Demo Features:
- 🎮 Interactive student profile configuration
- 📈 Real-time probability calculation
- 🎯 Risk band visualization
- 📋 Intervention recommendations
- 🔍 SHAP force plots and explanations
- ⚖️ Fairness dashboard

## 📊 Dataset

**Source**: Predict Students' Dropout and Academic Success (Portuguese higher education)

**Size**: 4,424 students, 36 features

**Features Include**:
- Demographic information (age, gender, nationality)
- Academic performance (grades, approved units)
- Economic factors (tuition status, scholarship)
- Enrollment details (course, attendance mode)

**Target Variable**: Dropout (1) vs Non-dropout (0: Graduate or Enrolled)

## 🔧 Development

### Training Pipeline
To retrain models from scratch:

```bash
# Run the complete pipeline
python baseline_dropout.py           # Step 1: Baseline
python dropout_models_step2.py       # Step 2: Model training
python dropout_risk_bands_step3.py   # Step 3: Risk bands
python dropout_interventions_step4.py # Step 4: Interventions
python dropout_shap_step5.py         # Step 5: SHAP explanations
python fairness_analysis_step6.py    # Step 6: Fairness analysis
```

### Adding New Features
1. Add feature preprocessing in `dropout_models_step2.py`
2. Update feature importance analysis
3. Modify SHAP explanation templates in `dropout_shap_step5.py`
4. Update the Streamlit UI in `app.py`

## ⚖️ Fairness Considerations

The system includes built-in fairness monitoring:

1. **Disparate Impact Analysis**: Intervention rates across demographic groups
2. **Error Rate Parity**: Equal false positive/negative rates
3. **Transparency**: All fairness metrics visible in the interface

**Note**: The fairness analysis is for institutional awareness and discussion, not automated decision-making.

## 🧪 Testing

### Unit Tests
```bash
# Run basic functionality tests
python -m pytest tests/ -v
```

### Model Validation
- 80/20 train-test split
- 5-fold cross-validation
- Out-of-time validation (when applicable)

## 🚨 Limitations & Ethical Considerations

### Limitations
1. **Data Specificity**: Trained on Portuguese higher education data
2. **Temporal Factors**: Does not capture real-time academic performance
3. **Causality**: Identifies correlations, not causal relationships
4. **Production Readiness**: Prototype stage, not deployment-ready

### Ethical Guidelines
1. **Human-in-the-loop**: Recommendations require advisor review
2. **Transparency**: All predictions are explainable
3. **Bias Monitoring**: Regular fairness audits required
4. **Student Consent**: Institutional policies must govern data usage
5. **Purpose Limitation**: Used only for support, not punitive measures

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Improvement
- Additional international datasets
- Real-time performance tracking
- Alternative banding strategies
- Enhanced fairness constraints
- Multi-institutional validation

## 📚 Citation

If you use this work in research, please cite:

(Unavailable)

## 📞 Contact

**Md Shakib Hasan**  
Faculty of Artificial Intelligence in Education  
Central China Normal University
mailto:shakib@mails.ccnu.edu.cn
Email: [Contact through GitHub](https://github.com/iskb-official)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The dataset is from UCI/Kaggle and subject to their respective terms.

---

**⚠️ Disclaimer**: This system is a research prototype. Institutions should conduct local validation, ethical review, and implement appropriate governance before real-world deployment.
