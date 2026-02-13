"""Custom dark theme for FPL Strategy Engine — fully de-Streamlit-ified."""

import streamlit as st

# Theme colors
COLORS = {
    'background': '#0a0a0b',
    'card': '#141416',
    'card_hover': '#1c1c1f',
    'border': '#2a2a2e',
    'accent': '#ef4444',
    'accent_hover': '#dc2626',
    'accent_glow': 'rgba(239, 68, 68, 0.15)',
    'text': '#b4b4b4',
    'text_muted': '#6b6b6b',
    'text_white': '#e8e8e8',
    'success': '#22c55e',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'surface': '#111113',
}

# Google Fonts import + full CSS overhaul
DARK_THEME_CSS = """
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Hide Streamlit sidebar and badges (keep toolbar/menu visible) ── */
    footer,
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .reportview-container .main footer,
    div[data-testid="stSidebarNav"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    section[data-testid="stSidebar"],
    div.viewerBadge_container__r5tak { display: none !important; }

    /* ── Root reset ── */
    .stApp {
        background: #0a0a0b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .main .block-container {
        max-width: 1320px;
        padding: 1.5rem 2rem 3rem 2rem;
    }
    
    /* ── Typography ── */
    p, span, label, div, .stMarkdown, input, select, textarea, button, td, th {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    p, span, label, .stMarkdown { color: #b4b4b4; font-size: 0.875rem; line-height: 1.6; }
    h1, h2, h3, h4, h5, h6 { color: #e8e8e8 !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    strong, b { color: #d4d4d4; }

    /* ── App header ── */
    .app-brand {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.75rem 0 0.5rem 0;
        border-bottom: 1px solid #1e1e21;
        margin-bottom: 1rem;
    }
    .app-brand-left {
        display: flex; align-items: baseline; gap: 0.75rem;
    }
    .app-brand-title {
        font-size: 1.25rem; font-weight: 700; color: #e8e8e8;
        letter-spacing: -0.03em; margin: 0;
    }
    .app-brand-badge {
        font-size: 0.6rem; font-weight: 600;
        background: rgba(239, 68, 68, 0.15); color: #ef4444;
        padding: 0.2rem 0.55rem; border-radius: 100px;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .app-brand-meta {
        font-size: 0.75rem; color: #4a4a4e; font-weight: 400;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Status pill ── */
    .status-bar {
        background: transparent; border: none;
        padding: 0; margin: 0 0 0.75rem 0;
        text-align: left; display: flex; gap: 0.75rem; flex-wrap: wrap;
    }
    .status-chip {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #141416; border: 1px solid #2a2a2e;
        padding: 0.3rem 0.7rem; border-radius: 6px;
        font-size: 0.72rem; color: #6b6b6b; font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .status-chip .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #22c55e; display: inline-block;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Section headings ── */
    .section-title {
        font-size: 0.8rem; font-weight: 600; color: #888;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 2rem 0 0.75rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e1e21;
    }

    /* ── Cards ── */
    .rule-card {
        background: #141416;
        border: 1px solid #2a2a2e;
        border-radius: 10px;
        padding: 1.25rem 1rem;
        margin: 0.35rem 0;
        text-align: center;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    .rule-card:hover {
        border-color: #3a3a3e;
        transform: translateY(-1px);
    }
    .rule-value {
        font-size: 1.75rem; font-weight: 700; color: #ef4444;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .rule-label {
        color: #6b6b6b; font-size: 0.72rem; font-weight: 500;
        margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.04em;
    }

    /* ── Tabs → looks like a segmented control / pill nav ── */
    .stTabs { margin-top: 0.25rem; }
    .stTabs [data-baseweb="tab-list"] {
        background: #111113;
        border: 1px solid #1e1e21;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6b6b6b;
        font-size: 0.72rem;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
        letter-spacing: -0.01em;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #b4b4b4;
        background: rgba(255,255,255,0.03);
    }
    .stTabs [aria-selected="true"] {
        background: #ef4444 !important;
        color: #fff !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Buttons ── */
    .stButton > button {
        background: #ef4444; color: #fff;
        border: none; border-radius: 8px;
        padding: 0.55rem 1.5rem;
        font-size: 0.8rem; font-weight: 600;
        letter-spacing: -0.01em;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
    }
    .stButton > button:hover {
        background: #dc2626;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Toggle / Switch ── */
    [data-testid="stToggle"] label span {
        font-size: 0.78rem !important; color: #6b6b6b !important; font-weight: 500 !important;
    }
    
    /* ── Inputs ── */
    .stTextInput input, .stNumberInput input {
        background: #141416 !important; color: #e8e8e8 !important;
        border: 1px solid #2a2a2e !important; border-radius: 8px !important;
        font-size: 0.85rem !important; padding: 0.55rem 0.75rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.1) !important;
    }
    /* ── Select / MultiSelect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #141416 !important; color: #e8e8e8 !important;
        border: 1px solid #2a2a2e !important; border-radius: 8px !important;
        font-size: 0.85rem !important;
    }
    /* Force the baseweb select container to use proper flexbox so the
       dropdown arrow sits on the far right instead of stacking on the label */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        width: 100% !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child {
        margin-left: auto !important;
        flex-shrink: 0 !important;
    }
    /* Ensure the value text doesn't push the arrow off-screen */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        padding-right: 0.5rem !important;
    }
    /* Prevent parent column constraints from collapsing the select */
    [data-testid="column"] .stSelectbox,
    [data-testid="column"] .stMultiSelect {
        min-width: 0 !important;
        width: 100% !important;
    }
    .stSlider label { color: #e8e8e8 !important; font-size: 0.8rem !important; }
    .stCheckbox label span { font-size: 0.8rem !important; }

    /* ── Metrics → mono style ── */
    [data-testid="stMetricValue"] {
        color: #e8e8e8 !important; font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem !important; letter-spacing: -0.03em;
    }
    [data-testid="stMetricLabel"] {
        color: #6b6b6b !important; font-size: 0.7rem !important;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stMetric"] {
        background: #141416; border: 1px solid #2a2a2e;
        border-radius: 10px; padding: 1rem !important;
    }

    /* ── DataFrames → sleek table ── */
    [data-testid="stDataFrame"],
    .stDataFrame {
        background: #111113; border-radius: 10px;
        border: 1px solid #1e1e21; overflow: hidden;
    }
    [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        border-radius: 10px;
    }
    /* Table header */
    [data-testid="stDataFrame"] th {
        background: #141416 !important; color: #6b6b6b !important;
        font-size: 0.7rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.05em;
        border-bottom: 1px solid #2a2a2e !important;
    }
    /* Table cells */
    [data-testid="stDataFrame"] td {
        font-size: 0.8rem !important; color: #b4b4b4 !important;
        border-bottom: 1px solid #1a1a1d !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        background: #141416 !important;
        border: 1px solid #2a2a2e !important;
        border-radius: 8px !important;
        color: #b4b4b4 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stAlertContainer"] > div {
        border-radius: 8px;
    }

    /* ── Expander ── */
    .stExpander {
        background: #141416; border: 1px solid #2a2a2e;
        border-radius: 10px !important;
    }
    /* Hide the native HTML disclosure triangle */
    [data-testid="stExpander"] summary {
        color: #e8e8e8; font-size: 0.82rem; font-weight: 500;
        padding: 0.75rem 1rem;
        list-style: none;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker { display: none; }
    [data-testid="stExpander"] summary::marker { display: none; content: ""; }
    /* Hide Streamlit's built-in SVG toggle icon */
    [data-testid="stExpander"] summary svg {
        display: none !important;
    }
    [data-testid="stExpander"] summary:hover { background: #1c1c1f; }
    .stExpander > div[data-testid="stExpanderDetails"] {
        border-top: 1px solid #1e1e21;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #ef4444 !important; }

    /* ── Plotly chart container ── */
    [data-testid="stPlotlyChart"] {
        background: #111113;
        border: 1px solid #1e1e21;
        border-radius: 10px;
        padding: 0.5rem;
        overflow: hidden;
    }

    /* ── Dividers ── */
    hr { border-color: #1e1e21 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0b; }
    ::-webkit-scrollbar-thumb { background: #2a2a2e; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a3e; }

    /* ── Custom checkbox / radio restyle ── */
    .stCheckbox [data-testid="stCheckbox"] {
        accent-color: #ef4444;
    }
    .stRadio label { font-size: 0.82rem !important; }

    /* ── Reduce Streamlit vertical bloat ── */
    .element-container { margin-bottom: 0.25rem; }
    [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    [data-testid="column"] { padding: 0 0.4rem; }

    /* ── Settings bar ── */
    .settings-bar {
        display: flex; align-items: center; gap: 1.5rem;
        padding: 0.4rem 0; margin-bottom: 0.25rem;
    }
</style>
"""


def apply_theme():
    """Apply custom dark theme CSS to the Streamlit app."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = None):
    """Render the app header as a brand bar."""
    badge = f'<span class="app-brand-badge">{subtitle}</span>' if subtitle else ''
    st.markdown(f'''
    <div class="app-brand">
        <div class="app-brand-left">
            <span class="app-brand-title">{title}</span>
            {badge}
        </div>
        <span class="app-brand-meta">v2.0</span>
    </div>
    ''', unsafe_allow_html=True)


def render_status_bar(text: str):
    """Render status chips."""
    # Parse the text for parts
    parts = [p.strip() for p in text.split('|')]
    chips = ''
    for i, part in enumerate(parts):
        dot = '<span class="dot"></span>' if i == 0 else ''
        chips += f'<span class="status-chip">{dot}{part}</span>'
    st.markdown(f'<div class="status-bar">{chips}</div>', unsafe_allow_html=True)


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
