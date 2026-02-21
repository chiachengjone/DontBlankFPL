"""FPL Strategy Engine - Streamlit Application."""

import logging
import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

# Limit float display globally
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Core imports only — tab modules are lazy-loaded below to reduce cold-start time
from fpl_api import create_data_pipeline
from components.styles import apply_theme, render_header, render_status_bar, render_tab_header

# Page configuration
st.set_page_config(
    page_title="DontBlank | FPL Strategy Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply dark theme
apply_theme()

# ── Session State Defaults ──
# We initialize these default values to ensure the app has a stable starting state.
# These control everything from the active model view to the user's preferred settings.
_SESSION_DEFAULTS = {
    'injury_highlight': True,           # Visual cue for injured players
    'use_poisson_xp': True,             # Legacy toggle, kept for backward compatibility
    'active_models': ['ml', 'poisson', 'fpl'], # Default to showing all 3 models for maximum context
    'fpl_team_id': 0,                   # Default Team ID (0 means no team loaded)
    # User Preferences (persisted across reruns)
    'pref_weeks_ahead': 1,              # Default horizon for xP calculations
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
    """
    Load FPL data with smart caching.
    
    This function creates the data pipeline and fetches fresh data.
    It is cached for 5 minutes (300s) to prevent hammering the FPL API 
    and to speed up the user experience on page reloads.
    """
    try:
        fetcher, processor = create_data_pipeline()
        # Trigger the property access to pre-build the players DataFrame 
        # inside the cache. This ensures subsequent access is instant.
        _ = processor.players_df
        return fetcher, processor, None
    except Exception as e:
        return None, None, str(e)


def main():
    """
    Main application entry point.
    Orchestrates the UI flow: Header -> Data Load -> Settings -> Tabs.
    """
    
    # ── Header & Title ──
    render_header("DontBlank", "AI-powered insights for Fantasy Premier League 2025/26")
    
    # ── Data Loading Phase ──
    with st.spinner("Loading data..."):
        fetcher, processor, error = load_fpl_data()
    
    # Handle critical data load failures gracefully
    if error:
        st.error(f"Failed to load FPL data: {error}")
        st.info("Check your internet connection and refresh.")
        # Fallback: Try to use the last known good data if the API fails
        if 'players_df' in st.session_state:
            st.warning("Showing last cached data (may be stale).")
            players_df = st.session_state['players_df']
        else:
            if st.button("Retry"):
                st.cache_resource.clear()
                st.rerun()
            return
    
    if processor is None:
        st.error("Could not initialize data processor")
        return
    
    # ── Feature Engineering ──
    # We cache the 'engineered' dataframe (with Poisson xP, etc.) separately 
    # because it depends on the 'weeks_ahead' setting, which can change.
    _ENGINE_VERSION = "v2_fixture_quality"  # Bump to invalidate stale engineered-df caches
    try:
        weeks_ahead = st.session_state.pref_weeks_ahead
        cache_key = f"_engineered_df_{_ENGINE_VERSION}_{weeks_ahead}"
        
        # Evict old horizon caches to bound memory
        _prefix = f"_engineered_df_{_ENGINE_VERSION}_"
        for k in [k for k in st.session_state if k.startswith(_prefix) and k != cache_key]:
            del st.session_state[k]
        
        if cache_key in st.session_state:
            players_df = st.session_state[cache_key]
        else:
            # This is the heavy lifting: calculating xP for the next N gameweeks
            players_df = processor.get_engineered_features_df(weeks_ahead=weeks_ahead)
            st.session_state[cache_key] = players_df
            
        st.session_state.players_df = players_df
        
    except Exception as e:
        st.error(f"Error loading players: {e}")
        return
    
    # ── ML Model Initialization ──
    # We auto-run the ML predictions on the first load or whenever the global horizon changes.
    # This ensures ML always evaluates the actual future fixtures (e.g. 3 GWs) natively.
    current_ml_gws = st.session_state.get("ml_gws", None)
    _ML_CODE_VERSION = "v2_per_game"  # Bump to force regeneration on code changes
    
    if ("ml_predictions" not in st.session_state
            or current_ml_gws != weeks_ahead
            or st.session_state.get("_ml_code_ver") != _ML_CODE_VERSION):
        try:
            from ml_predictor import create_ml_pipeline
            predictor = create_ml_pipeline(players_df)
            
            # Predict for the exact number of gameweeks in the horizon
            predictions = predictor.predict_gameweek_points(n_gameweeks=weeks_ahead, use_ensemble=True)
            
            st.session_state["ml_predictions"] = predictions
            st.session_state["ml_predictor"] = predictor
            st.session_state["ml_gws"] = weeks_ahead
            st.session_state["_ml_code_ver"] = _ML_CODE_VERSION
            
            # Run cross-validation to get accuracy metrics
            cv_scores = predictor.cross_validate_predictions(n_splits=3)
            st.session_state["ml_cv_scores"] = cv_scores
            
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.debug("ML auto-init skipped: %s", exc)
            # If ML fails to auto-load, we just skip it. The user can retry 
            # in the ML tab later.

    # ── Status Bar ──
    try:
        gw = fetcher.get_current_gameweek()
        render_status_bar(f"GW {gw} LIVE | {len(players_df)} players | Updated just now")
    except Exception:
        render_status_bar(f"{len(players_df)} players loaded")
    
    # ── Global Controls (Team ID & Model Selection) ──
    set_col1, set_col2, set_col3 = st.columns(3, vertical_alignment="center")
    
    # Column 1: Team ID Input
    with set_col1:
        with st.form("team_id_form", border=False):
            team_id_input = st.number_input(
                "FPL Team ID",
                min_value=0,
                max_value=99_999_999,
                value=st.session_state.fpl_team_id,
                step=1,
                help="Enter your FPL Team ID to analyze your specific squad."
            )
            submitted = st.form_submit_button("Sync Team", width="stretch")
            if submitted:
                st.session_state.fpl_team_id = int(team_id_input)
                st.rerun()

        # Understat status check
        ustat_active = st.session_state.get('_understat_active', None)
        if ustat_active is False:
            st.warning("Understat offline - using FPL fallback")
            
    # Column 2: Active Model Toggles
    # This allows the user to mix-and-match which prediction models they trust.
    with set_col2:
        st.markdown("<div style='font-size:0.75rem;color:#86868b;margin-bottom:0.2rem;'>Active Models (Model xP)</div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            use_ml = st.checkbox("ML xP", value='ml' in st.session_state.active_models, key="global_ml", help="Machine Learning Model")
        with m_col2:
            use_poisson = st.checkbox("Poi", value='poisson' in st.session_state.active_models, key="global_poisson", help="Poisson xG/xA Model")
        with m_col3:
            use_fpl = st.checkbox("FPL xP", value='fpl' in st.session_state.active_models, key="global_fpl", help="Official FPL Predictions")
            
        # Logic to update the active_models list based on checkboxes
        new_active = []
        if use_ml: new_active.append('ml')
        if use_poisson: new_active.append('poisson')
        if use_fpl: new_active.append('fpl')
        
        # Guard clause: Don't allow deselecting everything
        if not new_active:
            new_active = st.session_state.active_models if st.session_state.active_models else ['fpl']
            st.warning("At least one model must be selected.")
            
        if new_active != st.session_state.active_models:
            st.session_state.active_models = new_active
            st.session_state.use_poisson_xp = 'poisson' in new_active
            # Clear caches that depend on model selection (like Consensus EP)
            for key in list(st.session_state.keys()):
                if key.startswith('_cep_'):
                    del st.session_state[key]
            st.rerun()
            
    # Column 3: Global Visual Toggles
    with set_col3:
        st.session_state.injury_highlight = st.toggle(
            "Injury highlights",
            value=st.session_state.injury_highlight,
            help="Visually flag injured/doubtful players in tables"
        )
    
    # ── Main Navigation Tabs ──
    # Define the core structure of the application
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
    
    # Load each tab's content
    # Tabs are lazy-imported to reduce cold-start time.
    # .copy() is only used for tabs that mutate the DataFrame in-place.
    
    with tab0:
        render_tab_header("Dashboard", "At-a-glance overview of your gameweek — key metrics, top picks and alerts")
        from tabs.dashboard import render_dashboard_tab
        render_dashboard_tab(processor, players_df)
    
    with tab1:
        render_tab_header("Strategy", "Explore the player landscape, fixture difficulty and ownership trends")
        from tabs.strategy import render_strategy_tab
        render_strategy_tab(processor, players_df)
    
    with tab2:
        render_tab_header("Squad Builder", "Optimise transfers and build your best XV within budget")
        from tabs.optimization import render_optimization_tab
        render_optimization_tab(processor, players_df, fetcher)
    
    with tab3:
        render_tab_header("Analytics", "Deep-dive into player data, differentials and value metrics")
        from tabs.analytics import render_analytics_tab
        render_analytics_tab(processor, players_df)
    
    with tab4:
        render_tab_header("Captain", "Compare captain candidates using Poisson, ML and FPL estimates")
        from tabs.captain_tab import render_captain_tab
        render_captain_tab(processor, players_df, fetcher)
    
    with tab5:
        render_tab_header("Teams", "Team-level analysis — defensive stats, xG and fixture runs")
        from tabs.team_analysis_tab import render_team_analysis_tab
        render_team_analysis_tab(processor, players_df)

    with tab6:
        render_tab_header("Prices", "Track price changes, net transfers and rise/fall predictions")
        from tabs.price_predictor_tab import render_price_predictor_tab
        render_price_predictor_tab(processor, players_df)
    
    with tab7:
        render_tab_header("Wildcard", "Plan your wildcard squad with full-season optimisation")
        from tabs.wildcard_tab import render_wildcard_tab
        render_wildcard_tab(processor, players_df.copy())
    
    with tab8:
        render_tab_header("History", "Review past gameweek scores and season trajectory")
        from tabs.history_tab import render_history_tab
        render_history_tab(processor, players_df, fetcher)
    
    with tab9:
        render_tab_header("Rival Scout", "Compare squads head-to-head and spot tactical differences")
        from tabs.rival import render_rival_tab
        render_rival_tab(processor, players_df)
    
    with tab10:
        render_tab_header("ML Predictions", "Ensemble machine learning forecasts with confidence intervals")
        from tabs.ml_tab import render_ml_tab
        render_ml_tab(processor, players_df.copy())
    
    with tab11:
        render_tab_header("Monte Carlo", "Stochastic simulations for risk analysis and upside potential")
        from tabs.montecarlo_tab import render_monte_carlo_tab
        render_monte_carlo_tab(processor, players_df.copy())
    
    with tab12:
        render_tab_header("Genetic Optimizer", "Evolutionary algorithm for exploring diverse squad solutions")
        from tabs.genetic_tab import render_genetic_tab
        render_genetic_tab(processor, players_df, fetcher)
    


if __name__ == "__main__":
    main()
