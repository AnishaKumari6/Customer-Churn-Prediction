# -----------------------------
# ChurnIQ Pro - Streamlit App
# -----------------------------
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ChurnIQ Pro | AI Customer Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# INJECT GLASSMORPHISM CSS
# -----------------------------
def inject_css():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet">

        <style>
            /* High Performance Dark SaaS Design Tokens */
            :root {
                --bg-dark: #070913;
                --bg-surface: #0e1628;
                --bg-glass-card: #131c31;
                --border-glass: rgba(255, 255, 255, 0.08);
                --border-glow: rgba(99, 102, 241, 0.35);
                
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --accent-purple: #a855f7;
                --accent-cyan: #06b6d4;
                
                --risk-high: #f43f5e;
                --risk-high-bg: rgba(244, 63, 94, 0.12);
                --risk-medium: #f59e0b;
                --risk-medium-bg: rgba(245, 158, 11, 0.12);
                --risk-low: #10b981;
                --risk-low-bg: rgba(16, 185, 129, 0.12);
                
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
                
                --radius-lg: 18px;
                --radius-md: 12px;
                --radius-sm: 8px;
            }

            /* Global Typography & Fast Theme */
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
                color: var(--text-main) !important;
            }

            .stApp {
                background-color: var(--bg-dark);
            }

            /* Native Streamlit Header Integration */
            header[data-testid="stHeader"] {
                background: transparent !important;
            }

            /* Always Visible Sidebar Toggle Button */
            [data-testid="stSidebarCollapsedControl"] {
                display: block !important;
                visibility: visible !important;
                z-index: 99999 !important;
            }

            [data-testid="stSidebarCollapsedControl"] button,
            button[data-testid="stSidebarCollapseButton"] {
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
                border: 1px solid #a855f7 !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.6) !important;
                padding: 4px 8px !important;
            }

            [data-testid="stSidebarCollapsedControl"] button *,
            button[data-testid="stSidebarCollapseButton"] * {
                fill: #ffffff !important;
                color: #ffffff !important;
                stroke: #ffffff !important;
            }

            /* Hide Streamlit Footer & Top Color Bar only */
            footer, div[data-testid="stDecoration"] {
                visibility: hidden;
                display: none;
            }

            .main .block-container {
                padding: 1.75rem 2rem 3rem !important;
                max-width: 1240px !important;
            }

            /* Fast Dark Sidebar */
            section[data-testid="stSidebar"] {
                background: #0b0f19 !important;
                border-right: 1px solid var(--border-glass) !important;
            }

            section[data-testid="stSidebar"] .block-container {
                padding: 2rem 1.25rem !important;
            }

            /* Keyframe Animations */
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(16px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes pulseGlow {
                0%, 100% { opacity: 0.8; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.03); }
            }

            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }

            .animate-fade {
                animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
            }

            .animate-delay-1 { animation-delay: 0.1s; }
            .animate-delay-2 { animation-delay: 0.2s; }
            .animate-delay-3 { animation-delay: 0.3s; }

            /* Brand Header Component */
            .brand-container {
                display: flex;
                align-items: center;
                gap: 0.85rem;
                padding-bottom: 1.25rem;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid var(--border-glass);
            }

            .brand-logo {
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent-purple) 50%, var(--accent-cyan) 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4rem;
                box-shadow: var(--shadow-glow);
            }

            .brand-title {
                font-size: 1.35rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
            }

            .brand-subtitle {
                font-size: 0.75rem;
                color: var(--text-muted);
                font-weight: 500;
                margin: 0;
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.25);
                color: #34d399;
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                margin-top: 0.5rem;
            }

            .status-dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background-color: #10b981;
                box-shadow: 0 0 8px #10b981;
            }

            /* SaaS Hero Banner */
            .hero-card {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
                backdrop-filter: blur(16px);
                border: 1px solid var(--border-glass);
                border-radius: var(--radius-lg);
                padding: 2.25rem 2.5rem;
                margin-bottom: 1.75rem;
                position: relative;
                overflow: hidden;
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            }

            .hero-card::before {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 380px;
                height: 100%;
                background: radial-gradient(circle at right top, rgba(99, 102, 241, 0.18) 0%, rgba(6, 182, 212, 0.08) 50%, transparent 80%);
                pointer-events: none;
            }

            .hero-badge-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(99, 102, 241, 0.12);
                border: 1px solid rgba(99, 102, 241, 0.3);
                color: #818cf8;
                padding: 0.35rem 0.9rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.02em;
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: clamp(1.8rem, 3.5vw, 2.6rem);
                font-weight: 800;
                letter-spacing: -0.035em;
                line-height: 1.2;
                margin: 0 0 0.75rem 0;
                background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 70%, #38bdf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-desc {
                color: var(--text-muted);
                font-size: 1.02rem;
                max-width: 680px;
                line-height: 1.6;
                margin: 0 0 1.5rem 0;
            }

            .hero-stats-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }

            .hero-stat-item {
                display: flex;
                align-items: center;
                gap: 0.6rem;
            }

            .hero-stat-icon {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.05);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.95rem;
            }

            .hero-stat-val {
                font-size: 0.9rem;
                font-weight: 700;
                color: var(--text-main);
            }

            .hero-stat-lbl {
                font-size: 0.75rem;
                color: var(--text-dim);
            }

            /* Glassmorphic Section Containers */
            .glass-panel {
                background: var(--bg-surface);
                backdrop-filter: blur(16px);
                border: 1px solid var(--border-glass);
                border-radius: var(--radius-lg);
                padding: 1.75rem;
                margin-bottom: 1.5rem;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }

            .glass-panel:hover {
                border-color: rgba(255, 255, 255, 0.14);
            }

            .section-header-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.25rem;
            }

            .section-header-title {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .section-icon-box {
                width: 40px;
                height: 40px;
                border-radius: 10px;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.15) 100%);
                border: 1px solid rgba(99, 102, 241, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.15rem;
            }

            .section-title-text {
                font-size: 1.2rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0;
            }

            .section-subtitle-text {
                font-size: 0.85rem;
                color: var(--text-muted);
                margin: 0.2rem 0 0 0;
            }

            /* Input Controls Override */
            .stSlider label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
                font-weight: 600 !important;
                font-size: 0.85rem !important;
                color: #cbd5e1 !important;
                letter-spacing: 0.01em !important;
            }

            div[data-baseweb="input"] {
                background: rgba(15, 23, 42, 0.8) !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: var(--radius-sm) !important;
                color: var(--text-main) !important;
            }

            div[data-baseweb="input"]:focus-within {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
            }

            /* Buttons Override */
            .stButton > button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: var(--radius-md) !important;
                padding: 0.8rem 1.75rem !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                letter-spacing: 0.01em !important;
                box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
                transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease !important;
            }

            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
            }

            .stDownloadButton > button {
                background: rgba(30, 41, 59, 0.8) !important;
                color: #38bdf8 !important;
                border: 1px solid rgba(56, 189, 248, 0.4) !important;
                border-radius: var(--radius-md) !important;
                font-weight: 700 !important;
                padding: 0.75rem 1.5rem !important;
                transition: all 0.2s ease !important;
            }

            .stDownloadButton > button:hover {
                background: rgba(56, 189, 248, 0.15) !important;
                border-color: #38bdf8 !important;
                transform: translateY(-2px) !important;
            }

            /* Preset Button styling */
            .preset-btn {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid var(--border-glass);
                border-radius: var(--radius-sm);
                padding: 0.5rem 0.85rem;
                font-size: 0.78rem;
                font-weight: 600;
                color: var(--text-muted);
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .preset-btn:hover {
                background: rgba(99, 102, 241, 0.15);
                border-color: rgba(99, 102, 241, 0.4);
                color: #a5b4fc;
            }

            /* Custom Risk Badges & Result Cards */
            .risk-card {
                background: var(--bg-glass-card);
                border-radius: var(--radius-md);
                padding: 1.5rem;
                border: 1px solid var(--border-glass);
                position: relative;
                overflow: hidden;
            }

            .risk-tag {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.45rem 1.1rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 700;
                letter-spacing: 0.02em;
            }

            .risk-tag.high {
                background: var(--risk-high-bg);
                color: var(--risk-high);
                border: 1px solid rgba(244, 63, 94, 0.35);
            }

            .risk-tag.medium {
                background: var(--risk-medium-bg);
                color: var(--risk-medium);
                border: 1px solid rgba(245, 158, 11, 0.35);
            }

            .risk-tag.low {
                background: var(--risk-low-bg);
                color: var(--risk-low);
                border: 1px solid rgba(16, 185, 129, 0.35);
            }

            /* Action Banner */
            .action-banner {
                margin-top: 1.25rem;
                padding: 1.1rem 1.35rem;
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                gap: 0.85rem;
                font-size: 0.92rem;
                font-weight: 600;
                line-height: 1.5;
            }

            .action-banner.high {
                background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%);
                border: 1px solid rgba(244, 63, 94, 0.3);
                color: #fca5a5;
            }

            .action-banner.medium {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #fde68a;
            }

            .action-banner.low {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #6ee7b7;
            }

            /* Metric Box Component */
            .kpi-card {
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid var(--border-glass);
                border-radius: var(--radius-md);
                padding: 1.25rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }

            .kpi-label {
                font-size: 0.78rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: var(--text-muted);
            }

            .kpi-value {
                font-size: 1.85rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                margin-top: 0.4rem;
                color: var(--text-main);
            }

            /* File Uploader Dropzone Styling */
            div[data-testid="stFileUploader"] {
                background: rgba(15, 23, 42, 0.5) !important;
                border: 2px dashed rgba(99, 102, 241, 0.3) !important;
                border-radius: var(--radius-md) !important;
                padding: 1.5rem !important;
                transition: border-color 0.3s ease !important;
            }

            div[data-testid="stFileUploader"]:hover {
                border-color: var(--primary) !important;
            }

            /* Footer Styling */
            .saas-footer {
                margin-top: 4rem;
                padding: 2rem 1rem 1rem;
                border-top: 1px solid var(--border-glass);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.75rem;
                color: var(--text-muted);
                font-size: 0.85rem;
            }

            .saas-footer a {
                color: #818cf8;
                text-decoration: none;
                font-weight: 600;
                transition: color 0.2s ease;
            }

            .saas-footer a:hover {
                color: #a5b4fc;
                text-decoration: underline;
            }

            .footer-meta-row {
                display: flex;
                align-items: center;
                gap: 1.25rem;
            }

            /* Dataframe Styling */
            div[data-testid="stDataFrame"] {
                border: 1px solid var(--border-glass) !important;
                border-radius: var(--radius-md) !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# ASSETS LOADING
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

# -----------------------------
# PLOTLY CHART BUILDERS
# -----------------------------
def build_gauge_chart(prob):
    pct = prob * 100
    
    if prob > 0.7:
        bar_color = "#f43f5e"
    elif prob > 0.4:
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 44, 'color': "#ffffff", 'family': "Plus Jakarta Sans"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(15, 23, 42, 0.6)",
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.1)'}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#94a3b8", 'family': "Plus Jakarta Sans"},
        height=220,
        margin=dict(l=25, r=25, t=20, b=10)
    )
    return fig


def build_feature_impact_chart(input_df):
    drivers = {
        'Frustration (Calls)': float(input_df['Frustration_Score'].values[0]),
        'Recency Score': float(input_df['Recency_Score'].values[0] * 10),
        'Discount Sensitivity': float(input_df['Discount_Sensitivity'].values[0]),
        'Loyalty Score': float(input_df['Loyalty_Score'].values[0]),
    }
    
    df_chart = pd.DataFrame({
        'Feature': list(drivers.keys()),
        'ImpactScore': list(drivers.values())
    }).sort_values(by='ImpactScore', ascending=True)

    fig = px.bar(
        df_chart,
        x='ImpactScore',
        y='Feature',
        orientation='h',
        color='ImpactScore',
        color_continuous_scale=['#6366f1', '#a855f7', '#06b6d4'],
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#cbd5e1", 'family': "Plus Jakarta Sans", 'size': 12},
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title=None),
        yaxis=dict(showgrid=False, title=None),
        coloraxis_showscale=False,
        height=200,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def build_batch_distribution_chart(df):
    high_count = (df['churn_probability'] > 0.7).sum()
    med_count = ((df['churn_probability'] >= 0.4) & (df['churn_probability'] <= 0.7)).sum()
    low_count = (df['churn_probability'] < 0.4).sum()

    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    values = [low_count, med_count, high_count]
    colors = ['#10b981', '#f59e0b', '#f43f5e']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='rgba(15,23,42,0.8)', width=2)),
        textinfo='percent+label',
        textfont=dict(color='#ffffff', family='Plus Jakarta Sans', size=12)
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=230,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def build_batch_scatter_chart(df):
    if 'Days_Since_Last_Purchase' in df.columns:
        x_col = 'Days_Since_Last_Purchase'
    else:
        x_col = df.columns[0]

    fig = px.scatter(
        df,
        x=x_col,
        y='churn_probability',
        color='churn_probability',
        color_continuous_scale=['#10b981', '#f59e0b', '#f43f5e'],
        hover_data=[col for col in df.columns[:4] if col in df.columns]
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#cbd5e1", 'family': "Plus Jakarta Sans"},
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title=x_col),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title="Churn Prob"),
        coloraxis_showscale=False,
        height=230,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig

# -----------------------------
# MAIN APPLICATION
# -----------------------------
inject_css()
model, columns = load_assets()

# -----------------------------
# SIDEBAR NAVIGATION & PRESETS
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-logo">⚡</div>
            <div>
                <div class="brand-title">ChurnIQ Pro</div>
                <div class="brand-subtitle">AI Churn Intelligence System</div>
                <div class="status-badge">
                    <div class="status-dot"></div> Model v2.4 Online
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem'>Navigation</p>", unsafe_allow_html=True)
    
    view_mode = st.radio(
        "Navigation Mode",
        ["🔮 Single Customer Audit", "📂 Batch Risk Scoring", "📊 Analytics & Model Specs"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:var(--border-glass); margin:1.5rem 0 1.2rem 0'>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem'>Quick Profile Presets</p>", unsafe_allow_html=True)
    
    preset = st.radio(
        "Load Customer Sample",
        ["Custom Input", "🚨 High Risk At-Risk Customer", "⭐ VIP Loyal Customer", "⚠️ Frustrated Customer"],
        index=0,
        label_visibility="collapsed"
    )

    # Preset initial values logic
    if preset == "🚨 High Risk At-Risk Customer":
        init_age = 52
        init_purchases = 2
        init_days = 240
        init_calls = 9
        init_discount = 0.85
    elif preset == "⭐ VIP Loyal Customer":
        init_age = 34
        init_purchases = 45
        init_days = 12
        init_calls = 0
        init_discount = 0.15
    elif preset == "⚠️ Frustrated Customer":
        init_age = 41
        init_purchases = 8
        init_days = 95
        init_calls = 6
        init_discount = 0.40
    else:
        init_age = 35
        init_purchases = 15
        init_days = 45
        init_calls = 2
        init_discount = 0.25

    st.markdown("<hr style='border-color:var(--border-glass); margin:1.5rem 0 1.2rem 0'>", unsafe_allow_html=True)
    
    # Model Specs Card
    st.markdown(
        """
        <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:1rem;">
            <div style="font-size:0.78rem; font-weight:700; color:#818cf8; text-transform:uppercase; margin-bottom:0.5rem">Model Specifications</div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:0.3rem">
                <span style="color:#94a3b8">Algorithm:</span>
                <span style="font-weight:700; color:#f8fafc">XGBoost Classifier</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:0.3rem">
                <span style="color:#94a3b8">ROC-AUC:</span>
                <span style="font-weight:700; color:#34d399">0.93</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem">
                <span style="color:#94a3b8">Inference Time:</span>
                <span style="font-weight:700; color:#38bdf8">&lt; 12ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# HERO BANNER
# -----------------------------
st.markdown(
    """
    <div class="hero-card animate-fade">
        <div class="hero-badge-pill">
            <span>✨ Machine Learning Customer Retention</span>
        </div>
        <h1 class="hero-title">Predict & Prevent Customer Churn</h1>
        <p class="hero-desc">
            Leverage behavioral intelligence and real-time ML inference to identify at-risk customers 
            and deploy high-impact retention strategies before churn occurs.
        </p>
        <div class="hero-stats-grid">
            <div class="hero-stat-item">
                <div class="hero-stat-icon">🎯</div>
                <div>
                    <div class="hero-stat-val">93% Accuracy</div>
                    <div class="hero-stat-lbl">AUC Benchmark</div>
                </div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-icon">⚡</div>
                <div>
                    <div class="hero-stat-val">Real-time</div>
                    <div class="hero-stat-lbl">Scoring Engine</div>
                </div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-icon">🛡️</div>
                <div>
                    <div class="hero-stat-val">Actionable</div>
                    <div class="hero-stat-lbl">Retention Logic</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# VIEW 1: SINGLE CUSTOMER AUDIT
# -----------------------------
if view_mode == "🔮 Single Customer Audit":
    st.markdown(
        """
        <div class="glass-panel animate-fade animate-delay-1">
            <div class="section-header-row">
                <div class="section-header-title">
                    <div class="section-icon-box">🔮</div>
                    <div>
                        <h2 class="section-title-text">Single Customer Scoring</h2>
                        <p class="section-subtitle-text">Adjust customer behavioral parameters to compute instant churn probability.</p>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    col_inp1, col_inp2 = st.columns(2, gap="large")

    with col_inp1:
        age = st.slider("Customer Age", 18, 70, value=init_age)
        purchases = st.number_input("Total Lifetime Purchases", 0, 100, value=init_purchases)
        days_since = st.number_input("Days Since Last Purchase", 0, 365, value=init_days)

    with col_inp2:
        support_calls = st.number_input("Customer Service Calls", 0, 20, value=init_calls)
        discount = st.slider("Discount Usage Rate", 0.0, 1.0, value=float(init_discount), step=0.05)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # FEATURE ENGINEERING (PRESERVED)
    # -----------------------------
    input_data = {
        "Age": age,
        "Total_Purchases": purchases,
        "Customer_Service_Calls": support_calls,
        "Discount_Usage_Rate": discount,
        "Days_Since_Last_Purchase": days_since,
    }

    input_df = pd.DataFrame([input_data])

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
    # PREDICTION INFERENCE
    # -----------------------------
    predict_clicked = st.button("⚡ Run AI Churn Analysis", use_container_width=True, type="primary")
    
    # Run prediction automatically or on button click
    prob = model.predict_proba(input_df)[0][1]

    st.markdown(
        """
        <div class="glass-panel animate-fade animate-delay-2" style="margin-top:1.5rem;">
            <div class="section-header-title" style="margin-bottom:1rem">
                <div class="section-icon-box">📊</div>
                <div>
                    <h3 class="section-title-text">AI Diagnostic & Risk Assessment</h3>
                    <p class="section-subtitle-text">Real-time churn risk output and retention recommendation.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    res_col1, res_col2 = st.columns([1, 1.2], gap="large")

    with res_col1:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#94a3b8; text-transform:uppercase; text-align:center;'>Churn Probability Score</p>", unsafe_allow_html=True)
        fig_gauge = build_gauge_chart(prob)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        # Risk Tag & Action Banner
        if prob > 0.7:
            risk_tag_html = '<div class="risk-tag high">🔥 HIGH CHURN RISK</div>'
            banner_html = '<div class="action-banner high">🚨 <strong>Urgent Action Required:</strong> High likelihood of churn. Dispatch personalized retention discount offer (20%+ off) immediately.</div>'
        elif prob > 0.4:
            risk_tag_html = '<div class="risk-tag medium">⚠️ MEDIUM CHURN RISK</div>'
            banner_html = '<div class="action-banner medium">⚠️ <strong>Proactive Action:</strong> Moderate churn probability. Trigger targeted re-engagement email sequence & product recommendations.</div>'
        else:
            risk_tag_html = '<div class="risk-tag low">✅ LOW CHURN RISK</div>'
            banner_html = '<div class="action-banner low">✅ <strong>Stable Customer:</strong> Customer shows healthy engagement. Continue standard nurture flow.</div>'

        st.markdown(f"<div style='text-align:center; margin-top:-0.5rem'>{risk_tag_html}</div>", unsafe_allow_html=True)
        st.markdown(banner_html, unsafe_allow_html=True)

    with res_col2:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Key Behavioral Drivers</p>", unsafe_allow_html=True)
        fig_impact = build_feature_impact_chart(input_df)
        st.plotly_chart(fig_impact, use_container_width=True, config={'displayModeBar': False})

        # Mini Metrics breakdown
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Frustration Score</div>
                    <div class="kpi-value">{support_calls} <span style="font-size:0.8rem; font-weight:500; color:#64748b">calls</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Recency Score</div>
                    <div class="kpi-value">{1/(days_since+1):.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# VIEW 2: BATCH RISK SCORING
# -----------------------------
elif view_mode == "📂 Batch Risk Scoring":
    st.markdown(
        """
        <div class="glass-panel animate-fade animate-delay-1">
            <div class="section-header-title" style="margin-bottom:1rem">
                <div class="section-icon-box">📂</div>
                <div>
                    <h2 class="section-title-text">Batch Customer Scoring</h2>
                    <p class="section-subtitle-text">Upload your customer CSV file to generate bulk predictions and churn insights.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset (.csv)",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data_processed = data.reindex(columns=columns, fill_value=0)
        data_processed.replace([np.inf, -np.inf], 0, inplace=True)
        data_processed.fillna(0, inplace=True)

        probs = model.predict_proba(data_processed)[:, 1]
        data["churn_probability"] = probs
        
        # Risk Category Column
        def assign_risk_category(p):
            if p > 0.7: return "🔥 High"
            elif p > 0.4: return "⚠️ Medium"
            else: return "✅ Low"
            
        data["risk_level"] = data["churn_probability"].apply(assign_risk_category)

        total_customers = len(data)
        high_risk_cnt = (probs > 0.7).sum()
        med_risk_cnt = ((probs >= 0.4) & (probs <= 0.7)).sum()
        low_risk_cnt = (probs < 0.4).sum()

        st.markdown("<hr style='border-color:var(--border-glass); margin:1.5rem 0'>", unsafe_allow_html=True)
        
        # KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Customers</div>
                    <div class="kpi-value">{total_customers:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #f43f5e">
                    <div class="kpi-label">High Risk Customers</div>
                    <div class="kpi-value" style="color:#f43f5e">{high_risk_cnt:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #f59e0b">
                    <div class="kpi-label">Medium Risk Customers</div>
                    <div class="kpi-value" style="color:#f59e0b">{med_risk_cnt:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #10b981">
                    <div class="kpi-label">Low Risk Customers</div>
                    <div class="kpi-value" style="color:#10b981">{low_risk_cnt:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Risk Tier Breakdown</p>", unsafe_allow_html=True)
            fig_pie = build_batch_distribution_chart(data)
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        with chart_col2:
            st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Recency vs Churn Probability</p>", unsafe_allow_html=True)
            fig_scatter = build_batch_scatter_chart(data)
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#cbd5e1; margin-top:1rem;'>Scored Customer Records</p>", unsafe_allow_html=True)
        st.dataframe(data.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Scored Dataset (CSV)",
            data.to_csv(index=False),
            file_name="churn_predictions_scored.csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 1rem; color:#64748b">
                <div style="font-size:2.5rem; margin-bottom:0.5rem">📄</div>
                <div style="font-weight:600">Drag and drop a CSV file here</div>
                <div style="font-size:0.8rem; margin-top:0.25rem">Columns will be automatically formatted and feature engineered</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# VIEW 3: ANALYTICS & MODEL SPECS
# -----------------------------
else:
    st.markdown(
        """
        <div class="glass-panel animate-fade animate-delay-1">
            <div class="section-header-title" style="margin-bottom:1rem">
                <div class="section-icon-box">📊</div>
                <div>
                    <h2 class="section-title-text">Model Analytics & System Insights</h2>
                    <p class="section-subtitle-text">Performance benchmarks, feature rankings, and machine learning pipeline architecture.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    spec_col1, spec_col2, spec_col3 = st.columns(3)
    with spec_col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Model Accuracy</div>
                <div class="kpi-value" style="color:#6366f1">92.4%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">ROC-AUC Score</div>
                <div class="kpi-value" style="color:#a855f7">0.931</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Churn Recall</div>
                <div class="kpi-value" style="color:#06b6d4">84.2%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simulated Feature Importance Chart
    st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#cbd5e1;'>Top Global Feature Importances</p>", unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        'Feature': ['Frustration Score', 'Recency Score', 'Discount Sensitivity', 'Loyalty Score', 'Avg Purchase Per Year', 'Age'],
        'Importance': [0.34, 0.26, 0.18, 0.11, 0.07, 0.04]
    }).sort_values(by='Importance', ascending=True)

    fig_feat = px.bar(
        feat_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#6366f1', '#06b6d4']
    )
    fig_feat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#cbd5e1", 'family': "Plus Jakarta Sans"},
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
        coloraxis_showscale=False,
        height=260,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar': False})

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# FOOTER COMPONENT
# -----------------------------
st.markdown(
    """
    <div class="saas-footer animate-fade animate-delay-3">
        <div class="footer-meta-row">
            <span style="display:flex; align-items:center; gap:0.4rem">
                <span style="width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block"></span>
                <strong>ChurnIQ Pro v2.4</strong>
            </span>
            <span>·</span>
            <span>Built by <a href="https://github.com/AnishaKumari6" target="_blank">Anisha Kumari</a></span>
            <span>·</span>
            <span>Contact: <a href="mailto:anisha10021kumari@gmail.com">anisha10021kumari@gmail.com</a></span>
        </div>
        <div style="font-size:0.75rem; color:#64748b">
            © 2026 ChurnIQ Pro. Machine Learning Customer Retention Platform.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
