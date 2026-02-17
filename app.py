"""
FPL Strategy Engine - Streamlit Application
A high-performance dashboard for Fantasy Premier League 2025/26 season.
"""

import logging
import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

# Limit float display globally
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Import from local modules
from fpl_api import create_data_pipeline
from components.styles import apply_theme, render_header, render_status_bar, render_tab_header
from tabs.dashboard import render_dashboard_tab
from tabs.strategy import render_strategy_tab
from tabs.optimization import render_optimization_tab
from tabs.rival import render_rival_tab
from tabs.analytics import render_analytics_tab
from tabs.ml_tab import render_ml_tab
from tabs.montecarlo_tab import render_monte_carlo_tab
from tabs.genetic_tab import render_genetic_tab
from tabs.captain_tab import render_captain_tab
from tabs.team_analysis_tab import render_team_analysis_tab

from tabs.price_predictor_tab import render_price_predictor_tab
from tabs.history_tab import render_history_tab
from tabs.wildcard_tab import render_wildcard_tab

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
    'use_poisson_xp': True,  # Legacy, keeping for compatibility during migration
    'active_models': ['ml', 'poisson', 'fpl'], # Default to all 3
    'fpl_team_id': 0,
    # Preferences
    'pref_weeks_ahead': 1,
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
        # Graceful degradation: show cached data if available
        if '_last_good_players_df' in st.session_state:
            st.warning("Showing last cached data (may be stale).")
            players_df = st.session_state['_last_good_players_df']
        else:
            if st.button("Retry"):
                st.cache_resource.clear()
                st.rerun()
            return
    
    if processor is None:
        st.error("Could not initialize data processor")
        return
    
    # Get players data (cached per weeks_ahead to avoid re-running Poisson on every rerun)
    try:
        weeks_ahead = st.session_state.pref_weeks_ahead
        cache_key = f"_engineered_df_{weeks_ahead}"
        if cache_key in st.session_state:
            players_df = st.session_state[cache_key]
        else:
            players_df = processor.get_engineered_features_df(weeks_ahead=weeks_ahead)
            st.session_state[cache_key] = players_df
        st.session_state.players_df = players_df
        # Store last-good copy for graceful degradation on next failure
        st.session_state['_last_good_players_df'] = players_df
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return
    
    # Auto-run ML predictions on first load
    if "ml_predictions" not in st.session_state:
        try:
            from ml_predictor import create_ml_pipeline
            predictor = create_ml_pipeline(players_df)
            predictions = predictor.predict_gameweek_points(n_gameweeks=1, use_ensemble=True)
            st.session_state["ml_predictions"] = predictions
            st.session_state["ml_predictor"] = predictor
            st.session_state["ml_gws"] = 1
            cv_scores = predictor.cross_validate_predictions(n_splits=3)
            st.session_state["ml_cv_scores"] = cv_scores
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.debug("ML auto-init skipped: %s", exc)
            # Silently fall back — ML tab button still available

    # Status bar + settings on same row
    try:
        gw = fetcher.get_current_gameweek()
        render_status_bar(f"GW {gw} LIVE | {len(players_df)} players | Updated just now")
    except Exception:
        render_status_bar(f"{len(players_df)} players loaded")
    
    # Settings row (compact) -- Team ID + Toggles
    set_col1, set_col2, set_col3 = st.columns(3, vertical_alignment="center")
    with set_col1:
        with st.form("team_id_form", border=False):
            team_id_input = st.number_input(
                "FPL Team ID",
                min_value=0,
                max_value=99_999_999,
                value=st.session_state.fpl_team_id,
                step=1,
                help="Enter your FPL Team ID. Used by Squad Builder and Monte Carlo."
            )
            submitted = st.form_submit_button("Sync Team", width="stretch")
            if submitted:
                st.session_state.fpl_team_id = int(team_id_input)
                st.rerun()

        # Understat warning only when offline
        ustat_active = st.session_state.get('_understat_active', None)
        if ustat_active is False:
            st.warning("Understat offline - using FPL fallback")
    with set_col2:
        st.markdown("<div style='font-size:0.75rem;color:#86868b;margin-bottom:0.2rem;'>Active Models (Model xP)</div>", unsafe_allow_html=True)
        # Using checkboxes in a nested column layout for a clean look
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            use_ml = st.checkbox("ML xP", value='ml' in st.session_state.active_models, key="global_ml", help="ML Prediction")
        with m_col2:
            use_poisson = st.checkbox("Poi", value='poisson' in st.session_state.active_models, key="global_poisson", help="Poisson xP")
        with m_col3:
            use_fpl = st.checkbox("FPL xP", value='fpl' in st.session_state.active_models, key="global_fpl", help="FPL Raw xP")
            
        # Update active_models list
        new_active = []
        if use_ml: new_active.append('ml')
        if use_poisson: new_active.append('poisson')
        if use_fpl: new_active.append('fpl')
        
        # Enforce at least one model is always selected
        if not new_active:
            new_active = st.session_state.active_models if st.session_state.active_models else ['fpl']
            st.warning("At least one model must be selected.")
            
        if new_active != st.session_state.active_models:
            st.session_state.active_models = new_active
            # Also update legacy toggle for any old dependencies
            st.session_state.use_poisson_xp = 'poisson' in new_active
            # Clear consensus EP caches so they recalculate with new model selection
            for key in list(st.session_state.keys()):
                if key.startswith('_cep_'):
                    del st.session_state[key]
            st.rerun()
    with set_col3:
        st.session_state.injury_highlight = st.toggle(
            "Injury highlights",
            value=st.session_state.injury_highlight,
            help="Color-code rows by injury status"
        )
    
    # Navigation tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "Dashboard",
        "Strategy",
        "Squad Builder",
        "Analytics",
        "Captain",
        "Teams",
        "Prices",
        "Wildcard",
        "History",
        "Rival Scout",
        "ML Predict",
        "Monte Carlo",
        "Genetic",
    ])
    
    with tab0:
        render_tab_header("Dashboard", "At-a-glance overview of your gameweek — key metrics, top picks and alerts")
        render_dashboard_tab(processor, players_df.copy())
    
    with tab1:
        render_tab_header("Strategy", "Explore the player landscape, fixture difficulty and ownership trends")
        render_strategy_tab(processor, players_df.copy())
    
    with tab2:
        render_tab_header("Squad Builder", "Optimise transfers and build your best XV within budget")
        render_optimization_tab(processor, players_df.copy(), fetcher)
    
    with tab3:
        render_tab_header("Analytics", "Deep-dive into player data, differentials and value metrics")
        render_analytics_tab(processor, players_df.copy())
    
    with tab4:
        render_tab_header("Captain", "Compare captain candidates using Poisson, ML and FPL estimates")
        render_captain_tab(processor, players_df.copy(), fetcher)
    
    with tab5:
        render_tab_header("Teams", "Team-level analysis — defensive stats, xG and fixture runs")
        render_team_analysis_tab(processor, players_df.copy())

    with tab6:
        render_tab_header("Prices", "Track price changes, net transfers and rise/fall predictions")
        render_price_predictor_tab(processor, players_df.copy())
    
    with tab7:
        render_tab_header("Wildcard", "Plan your wildcard squad with full-season optimisation")
        render_wildcard_tab(processor, players_df.copy())
    
    with tab8:
        render_tab_header("History", "Review past gameweek scores and season trajectory")
        render_history_tab(processor, players_df.copy(), fetcher)
    
    with tab9:
        render_tab_header("Rival Scout", "Compare squads head-to-head and spot tactical differences")
        render_rival_tab(processor, players_df.copy())
    
    with tab10:
        render_tab_header("ML Predictions", "Ensemble machine learning forecasts with confidence intervals")
        render_ml_tab(processor, players_df.copy())
    
    with tab11:
        render_tab_header("Monte Carlo", "Stochastic simulations for risk analysis and upside potential")
        render_monte_carlo_tab(processor, players_df.copy())
    
    with tab12:
        render_tab_header("Genetic Optimizer", "Evolutionary algorithm for exploring diverse squad solutions")
        render_genetic_tab(processor, players_df.copy(), fetcher)
    



if __name__ == "__main__":
    main()
