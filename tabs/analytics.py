"""Analytics tab for FPL Strategy Engine.

This module contains only the top-level orchestrator.
All sub-render helpers live in ``analytics_helpers.py``.
"""

import streamlit as st
import pandas as pd

from config import POSITION_COLORS
from utils.helpers import (
    safe_numeric, style_df_with_injuries, round_df,
    classify_ownership_column, normalize_name,
    calculate_consensus_ep, get_consensus_label,
)
from tabs.analytics_helpers import (
    format_with_rank,
    render_player_table,
    render_points_distribution,
    render_value_by_position,
    render_cbit_analysis,
    render_advanced_metrics,
    render_set_and_forget,
    render_expected_vs_actual,
    render_ownership_trends,
)

POS_COLORS = POSITION_COLORS


def render_analytics_tab(processor, players_df: pd.DataFrame):
    """Analytics tab - player discovery and advanced metrics."""
    
    # Filters - use session state defaults
    f1, f2, f3 = st.columns([1, 1, 1])
    
    with f1:
        pos_filter = st.selectbox(
            "Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'],
            index=['All', 'GKP', 'DEF', 'MID', 'FWD'].index(st.session_state.get('pref_position', 'All')),
            key="analytics_pos",
        )
    with f2:
        max_price = st.slider(
            "Max Price", 4.0, 15.0,
            value=st.session_state.get('pref_max_price', 15.0),
            step=0.5, key="analytics_price",
        )
    with f3:
        min_mins = st.slider(
            "Min Minutes", 0, 1000,
            value=st.session_state.get('pref_min_mins', 90),
            step=90, key="analytics_mins",
        )
        
    f4, f5, f6 = st.columns([1, 1, 1])
    with f4:
        current_horizon = st.session_state.get('pref_weeks_ahead', 1)
        
        active_models = st.session_state.active_models
        con_ep_label = get_consensus_label(active_models, current_horizon)
        
        # Labels must match the column names in render_player_table exactly
        ep_label = f"Poisson xP ({current_horizon}GW)" if current_horizon > 1 else "Poisson xP"
        fpl_ep_label = f"FPL xP x{current_horizon}" if current_horizon > 1 else "FPL xP"
        ml_ep_label = f"ML xP x{current_horizon}" if current_horizon > 1 else "ML xP"
        
        sort_options = [con_ep_label]
        if current_horizon > 1 and len(active_models) > 1:
            sort_options.append(f'Avg {con_ep_label}')
        sort_options.extend([ep_label, fpl_ep_label, ml_ep_label, 'Total Points', 'Threat Momentum', 'xNP', 'Price', 'Own%'])
        
        sort_col = st.selectbox(
            "Sort By",
            sort_options,
            key="analytics_sort",
        )
    with f5:
        tier_filter = st.selectbox(
            "Ownership Tier",
            ['All', 'Template', 'Popular', 'Enabler', 'Differential'],
            key="analytics_tier",
        )
    with f6:
        # Horizon slider - controls multi-gameweek xP
        horizon = st.slider(
            "xP Horizon (GWs)", 1, 5,
            value=current_horizon,
            help="Sum Poisson xP over the next N gameweeks using full Poisson engine",
            key="analytics_horizon"
        )
        if horizon != current_horizon:
            st.session_state.pref_weeks_ahead = horizon
            # Clear cached players_df to force recalculation in app.py
            if 'players_df' in st.session_state:
                del st.session_state.players_df
            st.rerun()
    
    search = st.text_input("Search player...", placeholder="Enter name", key="analytics_search")
    
    # Prepare data - Analytics always prefers Poisson data for a high-performance view
    df = players_df.copy()
    
    # Force use of Poisson EP for Analytics tab metrics
    if 'expected_points_poisson' in df.columns:
        df['expected_points'] = safe_numeric(df['expected_points_poisson'])
    elif 'expected_points' not in df.columns or df['expected_points'].isna().all():
        # Fallback to blended/FPL if Poisson specifically missing
        df['expected_points'] = safe_numeric(df.get('ep_next', pd.Series([2.0]*len(df))))
    else:
        df['expected_points'] = safe_numeric(df['expected_points'])
    
    # Standardize FPL EP for ranking and comparison
    df['ep_next_val'] = safe_numeric(df.get('ep_next', 0))
    
    # ML & Consensus EP - Fetch ML Predictions from session state
    if 'ml_predictions' in st.session_state:
        ml_preds = st.session_state['ml_predictions']
        # Use raw single-GW predictions; calculate_consensus_ep will scale by horizon
        df['ml_pred'] = df['id'].apply(
            lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else df.loc[df['id'] == pid, 'expected_points'].iloc[0] / max(horizon, 1) * 0.85 if len(df[df['id'] == pid]) > 0 else 2.0
        ).round(2)
    else:
        # Fallback if no ML data - expected_points is already cumulative for the horizon
        df['ml_pred'] = (df['expected_points'] * 0.85 + (safe_numeric(df.get('form', 0)) * 0.3 * horizon)).round(2)
        
    # Consensus is the average of Poisson (expected_points), FPL (ep_next_val * horizon), and ML (ml_ep)
    # Use centralized consensus calculator
    df = calculate_consensus_ep(df, st.session_state.active_models, horizon)
        
    # Calculate Differential & Value Metrics
    if 'differential_gain' not in df.columns:
        eo = safe_numeric(df['selected_by_percent'], 5).clip(lower=0.1)
        eo_frac = eo / 100.0
        import numpy as np
        eo_10k = (1.5 * np.power(eo_frac, 1.4) + 0.01).clip(0.01, 0.99)
        df['eo_top10k'] = (eo_10k * 100).round(1)
        df['differential_gain'] = (df['consensus_ep'] * (1 - eo_10k)).round(2)
        df['diff_roi'] = (df['differential_gain'] / safe_numeric(df['now_cost'], 5).clip(lower=4)).round(3)
    
    if 'differential_score' not in df.columns:
        df['differential_score'] = df['differential_gain']
    
    df['cbit_propensity'] = safe_numeric(df.get('cbit_propensity', pd.Series([0]*len(df))))
    if df['cbit_propensity'].sum() == 0 and 'clean_sheets' in df.columns:
        df.loc[df['position'] == 'DEF', 'cbit_propensity'] = safe_numeric(df['clean_sheets']) / 10
    
    df['xg_per_pound'] = safe_numeric(df.get('xg_per_pound', df['expected_points'] / safe_numeric(df['now_cost'], 5).clip(lower=4)))
    
    # ── FILTER DATASET BEFORE RANKING ──
    # We rank based on the filtered criteria (Pos, Price, Mins, Tier)
    # but BEFORE the search filter to keep ranks stable while typing.
    mask = pd.Series([True] * len(df), index=df.index)
    if pos_filter != 'All': 
        mask &= (df['position'] == pos_filter)
    
    # Ensure numeric for filter
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['minutes'] = safe_numeric(df['minutes'])
    mask &= (df['now_cost'] <= max_price)
    mask &= (df['minutes'] >= min_mins)
    
    # Add ownership tier before applying tier filter
    df['ownership_tier'] = classify_ownership_column(df)
    if tier_filter != 'All': 
        mask &= (df['ownership_tier'] == tier_filter)
    
    rank_df = df[mask].copy()
    
    # Calculate ranks (1 = highest)
    if not rank_df.empty:
        rank_df['p_rank'] = rank_df['expected_points'].rank(ascending=False, method='min').astype(int)
        rank_df['f_rank'] = rank_df['ep_next_val'].rank(ascending=False, method='min').astype(int)
        rank_df['m_rank'] = rank_df['ml_pred'].rank(ascending=False, method='min').astype(int)
        rank_df['c_rank'] = rank_df['consensus_ep'].rank(ascending=False, method='min').astype(int)
        
        if horizon > 1 and 'avg_consensus_ep' in rank_df.columns:
            rank_df['ac_rank'] = rank_df['avg_consensus_ep'].rank(ascending=False, method='min').astype(int)
        # Add total points rank
        # Ensure 'total_points' is present and numeric
        rank_df['total_points'] = pd.to_numeric(rank_df.get('total_points', 0), errors='coerce').fillna(0)
        rank_df['tp_rank'] = rank_df['total_points'].rank(ascending=False, method='min').astype(int)
        
        # Create display columns with bracketed ranks
        # Using subscripts to make the rank look secondary and "lighter" (proxy for transparency)
        # while keeping the table interactive in st.dataframe

        rank_df['expected_points_display'] = rank_df.apply(
            lambda r: format_with_rank(f"{r['expected_points']:.2f}", r['p_rank']), axis=1
        )
        rank_df['ep_next_display'] = rank_df.apply(
            lambda r: format_with_rank(f"{r['ep_next_val'] * horizon:.2f}", r['f_rank']), axis=1
        )
        rank_df['ml_pred_display'] = rank_df.apply(
            lambda r: format_with_rank(f"{r['ml_pred']:.2f}", r['m_rank']), axis=1
        )
        rank_df['consensus_ep_display'] = rank_df.apply(
            lambda r: format_with_rank(f"{r['consensus_ep']:.2f}", r['c_rank']), axis=1
        )
        if horizon > 1 and 'ac_rank' in rank_df.columns:
            rank_df['avg_consensus_ep_display'] = rank_df.apply(
                lambda r: format_with_rank(f"{r['avg_consensus_ep']:.2f}", r['ac_rank']), axis=1
            )
        
        rank_df['total_points_display'] = rank_df.apply(
            lambda r: format_with_rank(f"{int(r['total_points'])}", r['tp_rank']), axis=1
        )
    else:
        rank_df['expected_points_display'] = ""
        rank_df['ep_next_display'] = ""
        rank_df['ml_pred_display'] = ""
        rank_df['consensus_ep_display'] = ""
        rank_df['total_points_display'] = ""

    # Apply search filter to the ranked subset
    df = rank_df
    if search:
        query = normalize_name(search.lower().strip())
        # Ensure normalized names exist for filtering
        if 'first_normalized' not in df.columns:
            df['first_normalized'] = df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
            df['second_normalized'] = df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
            
        df = df[
            (df['first_normalized'].str.startswith(query, na=False)) |
            (df['second_normalized'].str.startswith(query, na=False))
        ]
    
    sort_map = {
        con_ep_label: 'consensus_ep',
        f'Avg {con_ep_label}': 'avg_consensus_ep',
        ep_label: 'expected_points',
        fpl_ep_label: 'ep_next_val',
        ml_ep_label: 'ml_pred',
        'Total Points': 'total_points',
        'Threat Momentum': 'threat_momentum',
        'xNP': 'differential_gain',
        'Price': 'now_cost',
        'Own%': 'selected_by_percent',
    }
    sort_by = sort_map.get(sort_col, 'consensus_ep')
    if sort_by in df.columns:
        df[sort_by] = safe_numeric(df[sort_by])
        # ascending=df[sort_by].name == 'now_cost' doesn't work well in lambda, just check
        asc = (sort_col == 'Price')
        df = df.sort_values(sort_by, ascending=asc)
    elif sort_by == 'consensus_ep': # Fallback
        df = df.sort_values('expected_points', ascending=False)
    
    # Display player table
    render_player_table(df, horizon=horizon)
    
    # Points distribution (Main view, not in expander)
    render_points_distribution(players_df, con_ep_label)
    
    # Value by position
    with st.expander("Value Analysis", expanded=False):
        render_value_by_position(players_df, con_ep_label)
    
    # CBIT analysis chart
    with st.expander("CBIT Analysis", expanded=False):
        render_cbit_analysis(players_df)
    
    # Advanced Metrics (Threat Momentum, EPPM, Matchup Quality)
    with st.expander("Advanced Metrics", expanded=False):
        render_advanced_metrics(players_df, con_ep_label)
    
    # Set & Forget finder
    with st.expander("Set & Forget Picks", expanded=False):
        render_set_and_forget(players_df, con_ep_label)
    
    # Expected vs Actual
    with st.expander("Expected vs Actual", expanded=False):
        render_expected_vs_actual(players_df, con_ep_label)
    
    # Ownership Trends (always at the bottom, outside expander)
    render_ownership_trends(players_df)
