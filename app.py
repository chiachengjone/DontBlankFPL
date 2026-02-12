"""
FPL Strategy Engine - Streamlit Application
A high-performance dashboard for Fantasy Premier League 2025/26 season.
"""

import streamlit as st
import pandas as pd

# Import from local modules
from fpl_api import create_data_pipeline
from components.styles import apply_theme, render_header, render_status_bar
from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.planning import render_planning_tab

# Page configuration
st.set_page_config(
    page_title="FPL Strategy Engine 2025/26",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply dark theme
apply_theme()

# Initialize session state for injury highlighting
if 'injury_highlight' not in st.session_state:
    st.session_state.injury_highlight = True


@st.cache_resource(ttl=300, show_spinner=False)
def load_fpl_data():
    """Load FPL data with caching."""
    try:
        fetcher, processor = create_data_pipeline()
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
    
    # Settings row (compact)
    st.session_state.injury_highlight = st.toggle(
        "Injury highlights",
        value=st.session_state.injury_highlight,
        help="Color-code rows by injury status"
    )
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Strategy", 
        "Squad Builder", 
        "Analytics", 
        "Rival Scout",
        "Planning"
    ])
    
    with tab1:
        render_strategy_tab(processor, players_df)
    
    with tab2:
        render_optimization_tab(processor, players_df, fetcher)
    
    with tab3:
        render_analytics_tab(processor, players_df)
    
    with tab4:
        render_rival_tab(processor, players_df)
    
    with tab5:
        render_planning_tab(processor, players_df)


if __name__ == "__main__":
    main()
