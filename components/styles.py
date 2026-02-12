"""Dark theme styling for FPL Strategy Engine."""

import streamlit as st

# Theme colors
COLORS = {
    'background': '#0d0d0d',
    'card': '#1a1a1a',
    'border': '#333',
    'accent': '#ef4444',
    'accent_hover': '#dc2626',
    'text': '#ccc',
    'text_muted': '#888',
    'text_white': '#fff',
    'success': '#22c55e',
    'warning': '#f59e0b',
    'danger': '#ef4444',
}

# Full dark theme CSS
DARK_THEME_CSS = """
<style>
    [data-testid="stSidebar"] { display: none; }
    .stApp { background: #0d0d0d; }
    .main-header { font-size: 2rem; font-weight: 600; color: #fff; text-align: center; padding: 1rem 0; }
    .sub-header { font-size: 0.9rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .section-title { font-size: 1.2rem; font-weight: 500; color: #fff; margin: 1.5rem 0 1rem 0; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .rule-card { background: #1a1a1a; border: 1px solid #333; padding: 1rem; margin: 0.5rem 0; text-align: center; }
    .rule-value { font-size: 1.5rem; font-weight: 600; color: #ef4444; }
    .rule-label { color: #888; font-size: 0.85rem; }
    .status-bar { background: #1a1a1a; border: 1px solid #333; padding: 0.5rem 1rem; text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1a1a; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { background: #ef4444 !important; color: #fff !important; }
    .stButton > button { background: #ef4444; color: #fff; border: none; padding: 0.5rem 1.5rem; }
    .stButton > button:hover { background: #dc2626; }
    [data-testid="stMetricValue"] { color: #fff; }
    [data-testid="stMetricLabel"] { color: #888; }
    .stTextInput input, .stNumberInput input { background: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }
    .stSelectbox > div > div { background: #1a1a1a !important; color: #fff !important; }
    .stSlider label { color: #fff !important; }
    p, span, label, .stMarkdown { color: #ccc; }
    h1, h2, h3, h4, h5, h6 { color: #fff !important; }
    .stDataFrame { background: #1a1a1a; }
    [data-testid="stDataFrame"] { background: #1a1a1a; }
    .stAlert { background: #1a1a1a !important; border: 1px solid #333 !important; color: #ccc !important; }
    .stExpander { background: #1a1a1a; border: 1px solid #333; }
    [data-testid="stExpander"] summary { color: #fff; }
</style>
"""


def apply_theme():
    """Apply dark theme CSS to the Streamlit app."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = None):
    """Render the app header."""
    st.markdown(f'<p class="main-header">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="sub-header">{subtitle}</p>', unsafe_allow_html=True)


def render_status_bar(text: str):
    """Render a status bar."""
    st.markdown(f'<div class="status-bar">{text}</div>', unsafe_allow_html=True)


def render_section_title(title: str):
    """Render a section title."""
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


def render_rule_card(value: str, label: str):
    """Render a rule card with value and label."""
    st.markdown(f'''
    <div class="rule-card">
        <div class="rule-value">{value}</div>
        <div class="rule-label">{label}</div>
    </div>
    ''', unsafe_allow_html=True)
