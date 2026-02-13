"""
FPL Strategy Engine - Streamlit Application
A high-performance dashboard for Fantasy Premier League 2025/26 season.
"""

import streamlit as st
import pandas as pd

# Limit float display globally
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Import from local modules
from fpl_api import create_data_pipeline
from components.styles import apply_theme, render_header, render_status_bar
from tabs.dashboard import render_dashboard_tab
from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.ml_tab import render_ml_tab
from tabs.montecarlo_tab import render_monte_carlo_tab
from tabs.genetic_tab import render_genetic_tab

# Page configuration
st.set_page_config(
    page_title="FPL Strategy Engine 2025/26",
    page_icon="FPL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply dark theme
apply_theme()

# ── Session State Defaults ──
_SESSION_DEFAULTS = {
    'injury_highlight': True,
    'fpl_team_id': 0,
    # Preferences
    'pref_weeks_ahead': 5,
    'pref_strategy': 'Balanced',
    'pref_max_price': 15.0,
    'pref_position': 'All',
    'pref_min_mins': 90,
    'pref_sort': 'Expected Points',
}

for key, default in _SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(ttl=300, show_spinner=False)
def load_fpl_data():
    """Load FPL data with caching. Only fetches from API once per 5 minutes."""
    try:
        fetcher, processor = create_data_pipeline()
        # Pre-build the players DataFrame during cache so subsequent
        # calls to processor.players_df are instant
        _ = processor.players_df
        return fetcher, processor, None
    except Exception as e:
        return None, None, str(e)


def main():
    """Main application."""
    
    # Header
    render_header("FPL Strategy Engine", "2025/26")
    
    # Load data
    with st.spinner("Loading data..."):
        fetcher, processor, error = load_fpl_data()
    
    if error:
        st.error(f"Failed to load FPL data: {error}")
        st.info("Check your internet connection and refresh.")
        if st.button("Retry"):
            st.cache_resource.clear()
            st.rerun()
        return
    
    if processor is None:
        st.error("Could not initialize data processor")
        return
    
    # Get players data
    try:
        players_df = processor.players_df.copy()
        st.session_state.players_df = players_df
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return
    
    # Status bar + settings on same row
    try:
        gw = fetcher.get_current_gameweek()
        render_status_bar(f"GW {gw} LIVE | {len(players_df)} players | Updated just now")
    except:
        render_status_bar(f"{len(players_df)} players loaded")
    
    # Settings row (compact) -- Team ID + Injury highlights
    settings_left, settings_mid, settings_right = st.columns([1, 1, 1])
    with settings_left:
        team_id_input = st.number_input(
            "FPL Team ID",
            min_value=0,
            max_value=99999999,
            value=st.session_state.fpl_team_id,
            step=1,
            key="header_team_id",
            help="Enter your FPL Team ID (find it in the URL of your team page). Used by Squad Builder and Monte Carlo."
        )
        st.session_state.fpl_team_id = team_id_input
    with settings_mid:
        pref_horizon = st.slider(
            "Default Horizon (GWs)",
            min_value=3, max_value=10,
            value=st.session_state.pref_weeks_ahead,
            key="header_horizon",
            help="Default planning horizon used across tabs"
        )
        st.session_state.pref_weeks_ahead = pref_horizon
    with settings_right:
        st.session_state.injury_highlight = st.toggle(
            "Injury highlights",
            value=st.session_state.injury_highlight,
            help="Color-code rows by injury status"
        )
    
    # Navigation tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard",
        "Strategy", 
        "Squad Builder", 
        "Analytics", 
        "Rival Scout",
        "ML Predictions",
        "Monte Carlo",
        "Genetic",
    ])
    
    with tab0:
        render_dashboard_tab(processor, players_df)
    
    with tab1:
        render_strategy_tab(processor, players_df)
    
    with tab2:
        render_optimization_tab(processor, players_df, fetcher)
    
    with tab3:
        render_analytics_tab(processor, players_df)
    
    with tab4:
        render_rival_tab(processor, players_df)
    
    with tab5:
        render_ml_tab(processor, players_df)
    
    with tab6:
        render_monte_carlo_tab(processor, players_df)
    
    with tab7:
        render_genetic_tab(processor, players_df)


if __name__ == "__main__":
    main()
