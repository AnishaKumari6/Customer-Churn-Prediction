import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css():
    """Phase 1: Global design system — typography, palette, spacing, animations."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

        <style>
            :root {
                --primary: #4F46E5;
                --primary-light: #818CF8;
                --primary-dark: #3730A3;
                --accent: #06B6D4;
                --success: #10B981;
                --warning: #F59E0B;
                --danger: #EF4444;
                --bg: #F1F5F9;
                --surface: #FFFFFF;
                --text: #0F172A;
                --text-muted: #64748B;
                --border: #E2E8F0;
                --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
                --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.08);
                --shadow-lg: 0 12px 40px rgba(79, 70, 229, 0.12);
                --radius: 16px;
                --radius-sm: 10px;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }

            .stApp {
                background: linear-gradient(160deg, #EEF2FF 0%, var(--bg) 40%, #F8FAFC 100%);
            }

            header[data-testid="stHeader"] { background: transparent !important; }

            .main .block-container {
                padding: 1.5rem 1.25rem 3rem !important;
                max-width: 1100px !important;
            }

            #MainMenu, footer, header { visibility: hidden; }

            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(18px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to   { opacity: 1; }
            }

            @keyframes pulse-soft {
                0%, 100% { transform: scale(1); }
                50%      { transform: scale(1.02); }
            }

            .animate-in {
                animation: fadeInUp 0.55s ease-out both;
            }

            .animate-in-delay-1 { animation-delay: 0.08s; }
            .animate-in-delay-2 { animation-delay: 0.16s; }
            .animate-in-delay-3 { animation-delay: 0.24s; }

            /* Phase 2: Hero */
            .hero {
                background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 55%, var(--accent) 100%);
                border-radius: var(--radius);
                padding: 2.25rem 2rem;
                margin-bottom: 1.75rem;
                box-shadow: var(--shadow-lg);
                color: white;
                position: relative;
                overflow: hidden;
            }

            .hero::before {
                content: '';
                position: absolute;
                top: -40%;
                right: -10%;
                width: 320px;
                height: 320px;
                background: rgba(255,255,255,0.08);
                border-radius: 50%;
            }

            .hero-badge {
                display: inline-block;
                background: rgba(255,255,255,0.18);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 999px;
                padding: 0.35rem 0.9rem;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.85rem;
            }

            .hero h1 {
                font-size: clamp(1.75rem, 4vw, 2.5rem);
                font-weight: 800;
                margin: 0 0 0.6rem 0;
                letter-spacing: -0.03em;
                line-height: 1.15;
            }

            .hero p {
                font-size: clamp(0.95rem, 2vw, 1.05rem);
                opacity: 0.92;
                margin: 0 0 1.25rem 0;
                max-width: 620px;
                line-height: 1.6;
            }

            .hero-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .hero-tag {
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 999px;
                padding: 0.4rem 0.85rem;
                font-size: 0.82rem;
                font-weight: 500;
            }

            /* Phase 3 & 5: Cards */
            .section-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 1.75rem 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: var(--shadow-md);
            }

            .section-card-header {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                margin-bottom: 0.35rem;
            }

            .section-icon {
                width: 42px;
                height: 42px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.25rem;
                flex-shrink: 0;
            }

            .section-icon.purple { background: #EEF2FF; }
            .section-icon.cyan   { background: #ECFEFF; }
            .section-icon.green  { background: #ECFDF5; }

            .section-card h2 {
                font-size: 1.25rem;
                font-weight: 700;
                color: var(--text);
                margin: 0;
                letter-spacing: -0.02em;
            }

            .section-card .subtitle {
                color: var(--text-muted);
                font-size: 0.9rem;
                margin: 0 0 1.25rem 0;
                line-height: 1.5;
            }

            /* Streamlit widget overrides */
            .stSlider label, .stNumberInput label, .stFileUploader label {
                font-weight: 600 !important;
                color: var(--text) !important;
                font-size: 0.875rem !important;
            }

            div[data-testid="stMetric"] {
                background: var(--bg);
                border: 1px solid var(--border);
                border-radius: var(--radius-sm);
                padding: 1rem 1.1rem !important;
                animation: fadeInUp 0.45s ease-out both;
            }

            div[data-testid="stMetric"] label {
                color: var(--text-muted) !important;
                font-size: 0.8rem !important;
                font-weight: 600 !important;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 1.65rem !important;
                font-weight: 800 !important;
                color: var(--text) !important;
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: var(--radius-sm) !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                letter-spacing: 0.01em;
                box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            }

            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
            }

            .stDownloadButton > button {
                background: var(--surface) !important;
                color: var(--primary) !important;
                border: 2px solid var(--primary) !important;
                border-radius: var(--radius-sm) !important;
                font-weight: 700 !important;
                transition: all 0.2s ease !important;
            }

            .stDownloadButton > button:hover {
                background: #EEF2FF !important;
            }

            /* Phase 4: Custom metric cards */
            .metric-card {
                background: var(--bg);
                border: 1px solid var(--border);
                border-radius: var(--radius-sm);
                padding: 1.1rem 1.15rem;
                animation: fadeInUp 0.45s ease-out both;
                height: 100%;
            }

            .metric-card .metric-label {
                color: var(--text-muted);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.35rem;
            }

            .metric-card .metric-value {
                font-size: 1.65rem;
                font-weight: 800;
                color: var(--text);
                line-height: 1.2;
            }

            .metric-card.high   .metric-value { color: var(--danger); }
            .metric-card.medium .metric-value { color: var(--warning); }
            .metric-card.low    .metric-value { color: var(--success); }
            .metric-card.prob   .metric-value { color: var(--primary); }

            /* Input panel visual grouping */
            .input-panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 0.25rem 1.25rem 1.25rem;
                margin-bottom: 1.25rem;
                box-shadow: var(--shadow-md);
            }

            .result-header {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--text);
                margin: 1.25rem 0 0.85rem 0;
                letter-spacing: -0.01em;
            }

            .risk-banner {
                border-radius: var(--radius-sm);
                padding: 1rem 1.15rem;
                margin-top: 1rem;
                font-weight: 500;
                font-size: 0.92rem;
                animation: fadeInUp 0.4s ease-out both;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .risk-high   { background: #FEF2F2; border-left: 4px solid var(--danger); color: #991B1B; }
            .risk-medium { background: #FFFBEB; border-left: 4px solid var(--warning); color: #92400E; }
            .risk-low    { background: #ECFDF5; border-left: 4px solid var(--success); color: #065F46; }

            /* Batch upload zone hint */
            .upload-hint {
                background: #F8FAFC;
                border: 2px dashed var(--border);
                border-radius: var(--radius-sm);
                padding: 1rem;
                text-align: center;
                color: var(--text-muted);
                font-size: 0.85rem;
                margin-bottom: 0.75rem;
            }

            /* Footer */
            .footer {
                text-align: center;
                padding: 1.5rem 1rem 0.5rem;
                color: var(--text-muted);
                font-size: 0.875rem;
                animation: fadeIn 0.6s ease-out both;
            }

            .footer a {
                color: var(--primary);
                font-weight: 600;
                text-decoration: none;
            }

            .footer a:hover { text-decoration: underline; }

            .footer-brand {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                margin-top: 0.35rem;
                font-weight: 600;
                color: var(--text);
            }

            /* Mobile */
            @media (max-width: 768px) {
                .main .block-container {
                    padding: 1rem 0.85rem 2rem !important;
                }

                .hero {
                    padding: 1.5rem 1.25rem;
                    border-radius: 12px;
                }

                .section-card {
                    padding: 1.25rem 1rem;
                    border-radius: 12px;
                }

                div[data-testid="column"] {
                    min-width: 100% !important;
                }

                div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                    font-size: 1.35rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Phase 2: Hero header."""
    st.markdown(
        """
        <div class="hero animate-in">
            <div class="hero-badge">ML-Powered Analytics</div>
            <h1>📊 Churn Prediction</h1>
            <p>Predict whether a customer is likely to churn based on behavioral
            and transactional data — then act before they leave.</p>
            <div class="hero-tags">
                <span class="hero-tag">🎯 Identify high-risk users</span>
                <span class="hero-tag">🛡️ Proactive retention</span>
                <span class="hero-tag">📈 Optimize marketing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, variant=""):
    """Phase 4: Custom animated metric card."""
    st.markdown(
        f"""
        <div class="metric-card {variant}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_banner(prob):
    """Phase 4: Styled risk alert."""
    if prob > 0.7:
        st.markdown(
            '<div class="risk-banner risk-high">🚨 High Risk Customer — Immediate retention action needed</div>',
            unsafe_allow_html=True,
        )
    elif prob > 0.4:
        st.markdown(
            '<div class="risk-banner risk-medium">⚠️ Medium Risk — Engage user with targeted offers</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-banner risk-low">✅ Low Risk — Customer is stable</div>',
            unsafe_allow_html=True,
        )


def render_footer():
    """Phase 5: Footer."""
    st.markdown(
        """
        <div class="footer">
            Built by <a href="https://github.com/AnishaKumari6">Anisha Kumari</a>
            &nbsp;·&nbsp; anisha10021kumari@gmail.com
            <div class="footer-brand">📊 Churn Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# INIT
# -----------------------------
inject_css()

@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns


model, columns = load_assets()

# -----------------------------
# HERO
# -----------------------------
render_hero()

# -----------------------------
# SINGLE USER PREDICTION
# -----------------------------
st.markdown(
    """
    <div class="section-card animate-in animate-in-delay-1">
        <div class="section-card-header">
            <div class="section-icon purple">🔮</div>
            <h2>Single User Prediction</h2>
        </div>
        <p class="subtitle">Enter customer attributes below to get an instant churn probability score.</p>
    </div>
    <div class="input-panel animate-in animate-in-delay-1">
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.slider("Age", 18, 70)
    purchases = st.number_input("Total Purchases", 0, 100)
    days_since = st.number_input("Days Since Last Purchase", 0, 365)

with col2:
    support_calls = st.number_input("Customer Service Calls", 0, 20)
    discount = st.slider("Discount Usage Rate", 0.0, 1.0)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CREATE INPUT DATA
# -----------------------------
input_data = {
    "Age": age,
    "Total_Purchases": purchases,
    "Customer_Service_Calls": support_calls,
    "Discount_Usage_Rate": discount,
    "Days_Since_Last_Purchase": days_since,
}

input_df = pd.DataFrame([input_data])

# -----------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# -----------------------------
input_df["Engagement_score"] = 0
input_df["Avg_Purchase_Per_Year"] = purchases / (1 + 1)
input_df["Value_Per_Purchase"] = 0
input_df["Recency_Score"] = 1 / (days_since + 1)
input_df["Abandonment_Impact"] = 0
input_df["Frustration_Score"] = support_calls
input_df["Loyalty_Score"] = purchases
input_df["Discount_Sensitivity"] = discount * purchases

input_df = input_df.reindex(columns=columns, fill_value=0)
input_df.replace([np.inf, -np.inf], 0, inplace=True)
input_df.fillna(0, inplace=True)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn", use_container_width=True, type="primary"):
    prob = model.predict_proba(input_df)[0][1]

    st.markdown('<p class="result-header">📊 Prediction Result</p>', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3, gap="medium")

    with colA:
        render_metric_card("Churn Probability", f"{prob:.0%}", "prob")

    if prob > 0.7:
        with colB:
            render_metric_card("Risk Level", "High 🔥", "high")
        with colC:
            render_metric_card("Recommended Action", "Give Discount")
    elif prob > 0.4:
        with colB:
            render_metric_card("Risk Level", "Medium ⚠️", "medium")
        with colC:
            render_metric_card("Recommended Action", "Send Email")
    else:
        with colB:
            render_metric_card("Risk Level", "Low ✅", "low")
        with colC:
            render_metric_card("Recommended Action", "No Action")

    render_risk_banner(prob)

# -----------------------------
# BATCH PREDICTION
# -----------------------------
st.markdown(
    """
    <div class="section-card animate-in animate-in-delay-2">
        <div class="section-card-header">
            <div class="section-icon cyan">📂</div>
            <h2>Batch Prediction</h2>
        </div>
        <p class="subtitle">Upload a CSV file to score multiple customers at once and download results.</p>
        <div class="upload-hint">Supported format: .csv &nbsp;·&nbsp; Columns are auto-aligned to the model</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = data.reindex(columns=columns, fill_value=0)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)

    probs = model.predict_proba(data)[:, 1]
    data["churn_probability"] = probs

    st.markdown(
        """
        <div class="section-card-header" style="margin-top:0.5rem">
            <div class="section-icon green">📋</div>
            <h2 style="font-size:1.05rem">Preview Results</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(data.head(), use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions",
        data.to_csv(index=False),
        file_name="churn_predictions.csv",
        use_container_width=True,
    )

# -----------------------------
# FOOTER
# -----------------------------
render_footer()
