"""Light Apple-esque theme for FPL Strategy Engine."""

import streamlit as st

# ── Colour palette ──
COLORS = {
    'background': '#f5f5f7',
    'card': '#ffffff',
    'card_hover': '#fafafa',
    'border': 'rgba(0,0,0,0.06)',
    'border_hover': 'rgba(0,0,0,0.12)',
    'accent': '#007AFF',
    'accent_hover': '#0062d6',
    'accent_glow': 'rgba(0,122,255,0.12)',
    'text': '#1d1d1f',
    'text_secondary': '#86868b',
    'text_muted': '#aeaeb2',
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'surface': '#ffffff',
}

DARK_THEME_CSS = """
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Keyframes ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Hide Streamlit chrome ── */
    footer,
    header, header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .reportview-container .main footer,
    div[data-testid="stSidebarNav"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    div.viewerBadge_container__r5tak { display: none !important; }

    /* ── Kill Streamlit blue decoration + vignette ── */
    [data-testid="stDecoration"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header {
        display: none !important;
        background: transparent !important;
    }
    /* Force all Streamlit wrapper/container backgrounds */
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main, .block-container, section[data-testid="stMain"] {
        background: #f5f5f7 !important;
    }


    /* ── Root ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: #1d1d1f;
    }

    .main .block-container {
        max-width: 1280px;
        padding: 1.5rem 2rem 3rem 2rem;
    }

    /* ── Typography ── */
    p, span, label, div, .stMarkdown, input, select, textarea, button, td, th {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    p, span, label, .stMarkdown { color: #1d1d1f; font-size: 0.875rem; line-height: 1.6; }
    h1, h2, h3, h4, h5, h6 { color: #1d1d1f !important; font-weight: 600 !important; letter-spacing: -0.025em; }
    strong, b { color: #1d1d1f; }

    /* ── Hero ── */
    .hero-section {
        padding: 2rem 0 1.25rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        animation: fadeInUp 0.5s ease-out;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #1d1d1f;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #86868b;
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 0.3rem;
    }
    .hero-row {
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 1rem;
    }
    .hero-left { display: flex; flex-direction: column; }
    .hero-right { display: flex; align-items: center; gap: 0.6rem; }
    .hero-badge {
        font-size: 0.6rem; font-weight: 600;
        background: rgba(0,122,255,0.1); color: #007AFF;
        padding: 0.25rem 0.65rem; border-radius: 100px;
        letter-spacing: 0.06em; text-transform: uppercase;
        border: 1px solid rgba(0,122,255,0.15);
    }
    .hero-version {
        font-size: 0.7rem; color: #aeaeb2; font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Status pills ── */
    .status-bar {
        padding: 0; margin: 0 0 0.75rem 0;
        display: flex; gap: 0.6rem; flex-wrap: wrap;
        animation: fadeInUp 0.5s ease-out 0.05s both;
    }
    .status-chip {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        padding: 0.3rem 0.7rem; border-radius: 8px;
        font-size: 0.72rem; color: #86868b; font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .status-chip .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #34C759; display: inline-block;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 0.75rem; font-weight: 600; color: #86868b;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 1.75rem 0 0.75rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }

    /* ── Tab header card ── */
    .tab-header {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.04);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        animation: fadeInUp 0.4s ease-out;
    }
    .tab-header-row {
        display: flex; align-items: center; gap: 0.75rem;
    }
    .tab-header-icon {
        font-size: 1.5rem; line-height: 1;
    }
    .tab-header-text h3 {
        margin: 0 !important; font-size: 1.1rem !important; font-weight: 600 !important; color: #1d1d1f !important;
        letter-spacing: -0.02em;
    }
    .tab-header-text p {
        margin: 0.15rem 0 0 0 !important;
        color: #86868b !important; font-size: 0.8rem !important; line-height: 1.4;
    }

    /* ── Cards ── */
    .rule-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.04);
        border-radius: 14px;
        padding: 1.25rem 1rem;
        margin: 0.35rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease, transform 0.15s ease;
        animation: fadeInUp 0.4s ease-out both;
    }
    .rule-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }
    .rule-value {
        font-size: 1.75rem; font-weight: 700; color: #007AFF;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .rule-label {
        color: #86868b; font-size: 0.72rem; font-weight: 500;
        margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.04em;
    }

    /* ── Tabs ── */
    .stTabs { margin-top: 0.25rem; }
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        color: #86868b;
        font-size: 0.72rem;
        font-weight: 500;
        border-radius: 9px;
        padding: 0.45rem 0.7rem;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1d1d1f;
        background: rgba(0,0,0,0.03);
    }
    .stTabs [aria-selected="true"] {
        background: #007AFF !important;
        color: #fff !important;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,122,255,0.25);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Buttons ── */
    .stButton > button {
        background: #007AFF; color: #fff;
        border: none; border-radius: 10px;
        padding: 0.55rem 1.5rem;
        font-size: 0.8rem; font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,122,255,0.2);
    }
    .stButton > button:hover {
        background: #0062d6;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,122,255,0.3);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Toggle ── */
    [data-testid="stToggle"] label span {
        font-size: 0.78rem !important; color: #1d1d1f !important; font-weight: 500 !important;
    }
    /* Toggle track */
    [data-testid="stToggle"] label > div {
        background-color: #c7c7cc !important;
        border-radius: 16px !important;
        min-width: 44px !important;
        height: 24px !important;
    }
    [data-testid="stToggle"] label > div[aria-checked="true"] {
        background-color: #34C759 !important;
    }
    /* Toggle thumb */
    [data-testid="stToggle"] label > div > div {
        background-color: #ffffff !important;
        border-radius: 50% !important;
        width: 20px !important;
        height: 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] { color: #1d1d1f !important; font-weight: 600 !important; }
    .stSlider [role="slider"] {
        background-color: #1d1d1f !important;
        border: 2px solid #fff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }
    /* Slider filled track */
    .stSlider div[data-baseweb="slider"] div[role="progressbar"] {
        background-color: #1d1d1f !important;
    }
    /* Slider base track */
    .stSlider div[data-baseweb="slider"] > div > div:first-child {
        background-color: #d1d1d6 !important;
    }
    /* Slider tick marks / value labels */
    .stSlider label { color: #1d1d1f !important; }
    .stSlider .stMarkdown p { color: #86868b !important; }

    /* ── Checkbox accent ── */
    .stCheckbox label span[data-testid="stCheckboxLabel"] {
        color: #1d1d1f !important;
    }
    .stCheckbox svg { fill: #007AFF !important; }



    /* ── Inputs ── */
    .stTextInput input, .stNumberInput input {
        background: #ffffff !important; color: #1d1d1f !important;
        border: 1px solid rgba(0,0,0,0.1) !important; border-radius: 10px !important;
        font-size: 0.85rem !important; padding: 0.55rem 0.75rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 3px rgba(0,122,255,0.12) !important;
    }

    /* ── Select / MultiSelect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #ffffff !important; color: #1d1d1f !important;
        border: 1px solid rgba(0,0,0,0.1) !important; border-radius: 10px !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        display: flex !important; flex-wrap: nowrap !important;
        align-items: center !important; width: 100% !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child {
        margin-left: auto !important; flex-shrink: 0 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:first-child {
        flex: 1 1 auto !important; min-width: 0 !important;
        overflow: hidden !important; text-overflow: ellipsis !important;
        white-space: nowrap !important; padding-right: 0.5rem !important;
    }
    [data-testid="column"] .stSelectbox,
    [data-testid="column"] .stMultiSelect {
        min-width: 0 !important; width: 100% !important;
    }

    .stSlider label { color: #1d1d1f !important; font-size: 0.8rem !important; }
    .stCheckbox label span { font-size: 0.8rem !important; }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        color: #1d1d1f !important; font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem !important; letter-spacing: -0.03em;
    }
    [data-testid="stMetricLabel"] {
        color: #86868b !important; font-size: 0.7rem !important;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.04);
        border-radius: 14px; padding: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease, transform 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"], .stDataFrame {
        background: #ffffff; border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.04); overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { border-radius: 12px; }
    [data-testid="stDataFrame"] th {
        background: #f5f5f7 !important; color: #86868b !important;
        font-size: 0.7rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(0,0,0,0.06) !important;
    }
    [data-testid="stDataFrame"] td {
        font-size: 0.8rem !important; color: #1d1d1f !important;
        border-bottom: 1px solid rgba(0,0,0,0.04) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: rgba(0,122,255,0.03) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        background: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        border-radius: 10px !important;
        color: #1d1d1f !important;
        font-size: 0.82rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* ── Expander ── */
    .stExpander {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    [data-testid="stExpander"] summary {
        color: #1d1d1f; font-size: 0.82rem; font-weight: 500;
        padding: 0.75rem 1rem;
        list-style: none;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker { display: none; }
    [data-testid="stExpander"] summary::marker { display: none; content: ""; }
    [data-testid="stExpander"] summary svg { display: none !important; }
    [data-testid="stExpander"] summary:hover { background: rgba(0,0,0,0.02); }
    .stExpander > div[data-testid="stExpanderDetails"] {
        border-top: 1px solid rgba(0,0,0,0.06);
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #007AFF !important; }

    /* ── Plotly ── */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 0.5rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Dividers ── */
    hr { border-color: rgba(0,0,0,0.06) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f5f5f7; }
    ::-webkit-scrollbar-thumb { background: #d1d1d6; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #007AFF; }

    /* ── Radio ── */
    .stRadio label { font-size: 0.82rem !important; }

    /* ── Spacing ── */
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
    """Apply the light Apple theme CSS."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_hero():
    """Render the hero landing section."""
    st.markdown('''
    <div class="hero-section">
        <div class="hero-row">
            <div class="hero-left">
                <h1 class="hero-title">FPL Strategy Engine</h1>
                <p class="hero-subtitle">AI-powered insights for Fantasy Premier League 2025/26</p>
            </div>
            <div class="hero-right">
                <span class="hero-badge">2025/26</span>
                <span class="hero-version">v2.0</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_header(title: str, subtitle: str = None):
    """Legacy header — redirects to hero."""
    render_hero()


def render_tab_header(title: str, description: str, icon: str = ""):
    """Render a consistent intro card at the top of every tab."""
    st.markdown(
        f'<p style="background:#fff;border:1px solid rgba(0,0,0,0.06);border-radius:14px;'
        f'padding:1.25rem 1.5rem;margin-bottom:1.25rem;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.04);animation:fadeInUp 0.4s ease-out;">'
        f'<span style="font-size:1.1rem;font-weight:600;color:#1d1d1f;letter-spacing:-0.02em;">{title}</span><br>'
        f'<span style="color:#86868b;font-size:0.8rem;line-height:1.4;">{description}</span></p>',
        unsafe_allow_html=True,
    )


def render_status_bar(text: str):
    """Render status chips (pipe-delimited)."""
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


def render_stat_card(value: str, label: str, color: str = '#1d1d1f', sub_text: str = None):
    """Render a stat/metric card."""
    sub_html = f'<div style="color:#86868b;font-size:0.75rem;margin-top:0.25rem;">{sub_text}</div>' if sub_text else ''
    st.markdown(f'''
    <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);border-radius:14px;padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04);transition:box-shadow 0.2s ease,transform 0.15s ease;" onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.08)';this.style.transform='translateY(-1px)'" onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.04)';this.style.transform='none'">
        <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;letter-spacing:0.04em;">{label}</div>
        <div style="color:{color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:-0.03em;margin-top:0.25rem;">{value}</div>
        {sub_html}
    </div>
    ''', unsafe_allow_html=True)


def render_divider():
    """Render a consistent divider."""
    st.markdown('<hr style="border:none;border-top:1px solid rgba(0,0,0,0.06);margin:1.5rem 0;">', unsafe_allow_html=True)


def render_subsection_title(title: str):
    """Render a subsection title."""
    st.markdown(f'<p style="color:#1d1d1f;font-size:1rem;font-weight:600;margin:1.5rem 0 0.75rem 0;">{title}</p>', unsafe_allow_html=True)


def render_fixture_badge(gw: int, opponent: str, venue: str, fdr: int):
    """Render a fixture badge with FDR coloring.

    Args:
        gw: Gameweek number
        opponent: Opponent short name
        venue: 'H' or 'A'
        fdr: Fixture difficulty rating (1-5)
    """
    fdr_colors = {
        1: '#34C759',
        2: '#4ade80',
        3: '#FF9500',
        4: '#FF3B30',
        5: '#dc2626',
    }
    color = fdr_colors.get(int(fdr), '#86868b')

    st.markdown(f'''
    <div style="background:{color};border-radius:10px;padding:0.5rem;text-align:center;">
        <div style="color:#fff;font-size:0.65rem;font-weight:500;">GW{gw}</div>
        <div style="color:#fff;font-size:0.9rem;font-weight:600;">{opponent}</div>
        <div style="color:rgba(255,255,255,0.8);font-size:0.65rem;">({venue})</div>
    </div>
    ''', unsafe_allow_html=True)

