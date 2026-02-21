"""Analytics tab for FPL Strategy Engine.

This module contains only the top-level orchestrator.
All sub-render helpers live in ``analytics_helpers.py``.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from config import POSITION_COLORS
from utils.helpers import (
    safe_numeric, style_df_with_injuries, round_df,
    classify_ownership_column, normalize_name,
    calculate_consensus_ep, get_consensus_label,
    evict_old_session_keys,
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
    with f6:
        # Horizon slider - LOCAL to Analytics tab only
        horizon = st.slider(
            "xP Horizon (GWs)", 1, 5,
            value=st.session_state.get('_analytics_horizon_val', 1),
            help="Sum xP over the next N gameweeks (only affects this table)",
            key="analytics_horizon"
        )
        st.session_state['_analytics_horizon_val'] = horizon

    with f4:
        active_models = st.session_state.active_models
        con_ep_label = get_consensus_label(active_models, horizon)
        
        # Labels must match the column names in render_player_table exactly
        ep_label = f"Poisson xP ({horizon}GW)" if horizon > 1 else "Poisson xP"
        fpl_ep_label = f"FPL xP x{horizon}" if horizon > 1 else "FPL xP"
        ml_ep_label = f"ML xP x{horizon}" if horizon > 1 else "ML xP"
        
        sort_options = [con_ep_label]
        if horizon > 1 and len(active_models) > 1:
            sort_options.append(f'Avg {con_ep_label}')
        sort_options.extend([ep_label, fpl_ep_label, ml_ep_label, 'Total Points', 'Threat Momentum', 'CBIT', 'Price', 'Own%'])
        
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
    
    search = st.text_input("Search player...", placeholder="Enter name", key="analytics_search")
    
    # Prepare data - Analytics always prefers Poisson data for a high-performance view
    df = players_df.copy()
    
    # ── Recalculate Poisson EP locally for THIS tab's horizon ──
    # The global players_df may have been computed for a different horizon,
    # so we re-run the Poisson engine here for the slider's value.
    _analytics_poisson_key = f"_analytics_poisson_{horizon}"
    evict_old_session_keys("_analytics_poisson_", _analytics_poisson_key)
    if horizon > 1 and _analytics_poisson_key not in st.session_state:
        try:
            from poisson_ep import calculate_poisson_ep_for_dataframe
            fixtures_df = processor.fixtures_df
            current_gw = processor.fetcher.get_current_gameweek()
            team_stats = st.session_state.get('_understat_team_stats', None)
            poisson_result = calculate_poisson_ep_for_dataframe(
                df, fixtures_df, current_gw, team_stats=team_stats, horizon=horizon
            )
            st.session_state[_analytics_poisson_key] = poisson_result[[
                c for c in ('expected_points_poisson', 'games_in_horizon', 'fixture_quality_factor')
                if c in poisson_result.columns
            ]].copy()
        except Exception:
            pass
    
    if _analytics_poisson_key in st.session_state:
        cached_poisson = st.session_state[_analytics_poisson_key]
        for col in cached_poisson.columns:
            df[col] = cached_poisson[col].values
    
    # Force use of Poisson EP for Analytics tab metrics
    if 'expected_points_poisson' in df.columns:
        df['expected_points'] = safe_numeric(df['expected_points_poisson'])
    elif 'expected_points' not in df.columns or df['expected_points'].isna().all():
        # Fallback to blended/FPL if Poisson specifically missing
        df['expected_points'] = safe_numeric(df.get('ep_next', pd.Series([2.0]*len(df))))
    else:
        df['expected_points'] = safe_numeric(df['expected_points'])
    
    # ML & Consensus EP - Fetch ML Predictions from session state (per-game values)
    if 'ml_predictions' in st.session_state:
        ml_preds = st.session_state['ml_predictions']
        # ML predictions are per-game. calculate_consensus_ep will scale
        # by games_in_horizon for the correct multi-GW total.
        df['ml_pred'] = df['id'].apply(
            lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else (df.loc[df['id'] == pid, 'expected_points'].iloc[0] * 0.85 if len(df[df['id'] == pid]) > 0 else 2.0)
        ).round(2)
    else:
        # Fallback if no ML data
        df['ml_pred'] = (df['expected_points'] * 0.85 + (safe_numeric(df.get('form', 0)) * 0.3)).round(2)
        
    # Consensus calculation — scales FPL & ML by actual games_in_horizon
    # (handles DGWs and BGWs), Poisson is already summed per-fixture.
    df = calculate_consensus_ep(df, st.session_state.active_models, horizon)
    
    # Use the properly scaled totals from consensus for display columns
    df['ep_next_val'] = safe_numeric(df.get('fpl_xp_total', safe_numeric(df.get('ep_next', 0)) * horizon))
    df['ml_pred'] = safe_numeric(df.get('ml_xp_total', df['ml_pred']))
        
    # Calculate Differential & Value Metrics (RECALCULATE for current models/horizon)
    eo = safe_numeric(df['selected_by_percent'], 5).clip(lower=0.1)
    eo_frac = eo / 100.0
    import numpy as np
    eo_10k = (1.5 * np.power(eo_frac, 1.4) + 0.01).clip(0.01, 0.99)
    df['eo_top10k'] = (eo_10k * 100).round(1)
    
    # Core RRI: xNP (Expected Net Points) = Model xP * (1 - EO)
    df['differential_gain'] = (df['consensus_ep'] * (1 - eo_10k)).round(2)
    df['xnp'] = df['differential_gain'] # Standardize naming for helpers
    df['diff_roi'] = (df['differential_gain'] / safe_numeric(df['now_cost'], 5).clip(lower=4)).round(3)
    
    # Recalculate Engineered Differential Score using updated components
    low_own_bonus = np.where(eo < 10, (10 - eo) / 10, 0)
    df['engineered_diff'] = (
        df['differential_gain'] * 0.35 +
        safe_numeric(df.get('eppm', 0)) * 0.25 +
        safe_numeric(df.get('threat_momentum', 0)) * 0.20 +
        safe_numeric(df.get('matchup_quality', 0)) * 0.10 +
        low_own_bonus * 0.10
    ).round(2)
    
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

        # No longer creating display columns with ranks to allow numeric sorting
        # Ranks are still calculated in rank_df if we want to display them separately, but for now we rely on sorting.
        pass
    else:
        pass

    # Apply search filter to the ranked subset
    filtered_df = rank_df
    if search:
        query = normalize_name(search.lower().strip())
        # Ensure normalized names exist for filtering
        if 'first_normalized' not in filtered_df.columns:
            filtered_df['first_normalized'] = filtered_df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
            filtered_df['second_normalized'] = filtered_df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
            
        filtered_df = filtered_df[
            (filtered_df['first_normalized'].str.startswith(query, na=False)) |
            (filtered_df['second_normalized'].str.startswith(query, na=False))
        ]
    
    sort_map = {
        con_ep_label: 'consensus_ep',
        f'Avg {con_ep_label}': 'avg_consensus_ep',
        ep_label: 'expected_points',
        fpl_ep_label: 'ep_next_val',
        ml_ep_label: 'ml_pred',
        'Total Points': 'total_points',
        'Threat Momentum': 'threat_momentum',
        'Price': 'now_cost',
        'Own%': 'selected_by_percent',
        'CBIT': 'cbit_score'
    }
    sort_by = sort_map.get(sort_col, 'consensus_ep')
    if sort_by in filtered_df.columns:
        filtered_df[sort_by] = safe_numeric(filtered_df[sort_by])
        # ascending=df[sort_by].name == 'now_cost' doesn't work well in lambda, just check
        asc = (sort_col == 'Price')
        filtered_df = filtered_df.sort_values(sort_by, ascending=asc)
    elif sort_by == 'consensus_ep': # Fallback
        filtered_df = filtered_df.sort_values('expected_points', ascending=False)
    
    # Display player table
    render_player_table(filtered_df, horizon=horizon)
    
    # Points distribution (Main view, not in expander)
    render_points_distribution(df, con_ep_label)
    
    # Value by position
    with st.expander("Value Analysis", expanded=False):
        render_value_by_position(df, con_ep_label)
    
    # CBIT analysis chart
    with st.expander("CBIT Analysis", expanded=False):
        render_cbit_analysis(df)
    
    # Advanced Metrics (Threat Momentum, EPPM, Matchup Quality)
    with st.expander("Advanced Metrics", expanded=False):
        render_advanced_metrics(df, con_ep_label)
    
    # Set & Forget finder
    with st.expander("Set & Forget Picks", expanded=False):
        render_set_and_forget(df, con_ep_label)
    
    # Expected vs Actual
    with st.expander("Expected vs Actual", expanded=False):
        render_expected_vs_actual(df, con_ep_label)
    
    with st.expander("Player Role Analysis (xG vs xA)", expanded=False):
        st.caption("Detailed breakdown of Goal Threat vs Creativity.")
        
        # 1. Filters Setup
        rf1, rf2, rf3, rf4 = st.columns([1, 1, 1, 1])
        with rf1:
            role_team = st.multiselect("Team", sorted(df['team_name'].unique()), key="role_team")
        with rf2:
            # We can't know roles until we classify, but we can pre-define the known roles
            role_options = ["Double Threat", "Goal Threat", "Creative Hub", "Rotation/Low"]
            role_filter = st.multiselect("Role", role_options, key="role_select")
        with rf3:
            role_sort = st.selectbox("Sort By", ["xG", "xA", "xP", "Price"], key="role_sort")
        with rf4:
            role_search = st.text_input("Search", key="role_search")
            
        rf5, rf6 = st.columns([1, 1])
        with rf5:
            role_min_mins = st.slider("Min Minutes", 0, 3000, 500, 100, key="role_mins")
        with rf6:
            role_max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="role_price")

            
        # 2. Data Preparation
        # Use Understat data if available
        xg_col = 'us_xG_per90' if 'us_xG_per90' in df.columns else 'expected_goals_per_90'
        xa_col = 'us_xA_per90' if 'us_xA_per90' in df.columns else 'expected_assists_per_90'
        
        role_df = df[
            (df['minutes'] >= role_min_mins) & 
            (df['now_cost'] <= role_max_price) &
            (df['position'].isin(['DEF', 'MID', 'FWD']))
        ].copy()
        
        if role_team:
            role_df = role_df[role_df['team_name'].isin(role_team)]
        
        if role_search:
            query = normalize_name(role_search.lower().strip())
            role_df = role_df[role_df['web_name'].apply(lambda x: normalize_name(str(x).lower())).str.contains(query, na=False)]

        role_df[xg_col] = safe_numeric(role_df[xg_col])
        role_df[xa_col] = safe_numeric(role_df[xa_col])
        role_df['consensus_ep'] = safe_numeric(role_df['consensus_ep'])
        
        if not role_df.empty:
            avg_xg = role_df[xg_col].mean()
            avg_xa = role_df[xa_col].mean()
            
            def classify_threat(row):
                high_g = row[xg_col] > avg_xg * 1.2
                high_a = row[xa_col] > avg_xa * 1.2
                if high_g and high_a: return "Double Threat"
                if high_g: return "Goal Threat"
                if high_a: return "Creative Hub"
                return "Rotation/Low"
            
            role_df['Role'] = role_df.apply(classify_threat, axis=1)
            
            # Apply Role Filter
            if role_filter:
                role_df = role_df[role_df['Role'].isin(role_filter)]

            # Calculate Ranks for xG and xA (within filtered set)
            if not role_df.empty:
                role_df['xg_rank'] = role_df[xg_col].rank(ascending=False, method='min').astype(int)
                role_df['xa_rank'] = role_df[xa_col].rank(ascending=False, method='min').astype(int)
                
                # Format Columns
                # Format Columns - Use raw numeric for sorting
                role_df['ep_val'] = role_df['consensus_ep']

                # 3. Table (Appears First)
                st.markdown("#### Role Breakdown")
                
                sort_map = {"xG": xg_col, "xA": xa_col, "xP": "consensus_ep", "Price": "now_cost"}
                sort_col = sort_map.get(role_sort, xg_col)
                asc = (role_sort == "Price")
                
                role_disp = role_df[['web_name', 'team_name', 'position', 'Role', xg_col, xa_col, 'consensus_ep', 'now_cost']].copy()
                role_disp.columns = ['Player', 'Team', 'Pos', 'Role', 'xG/90', 'xA/90', con_ep_label, 'Price']
                
                # Sort the source to determine order
                sorted_idx = role_df.sort_values(sort_col, ascending=asc).index
                role_disp = role_disp.loc[sorted_idx]
                
                st.dataframe(
                    style_df_with_injuries(role_disp, players_df),
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="£%.1fm"),
                        "xG/90": st.column_config.NumberColumn(format="%.2f"),
                        "xA/90": st.column_config.NumberColumn(format="%.2f"),
                        con_ep_label: st.column_config.NumberColumn(format="%.2f"),
                    }
                )

                # 4. Scatter Plot (Appears Below)
                st.markdown("#### Visualization")
                fig = px.scatter(
                    role_df,
                    x=xa_col,
                    y=xg_col,
                    color='Role',
                    hover_name='web_name',
                    hover_data=['team_name', 'position', 'now_cost'],
                    labels={xg_col: "xG per 90", xa_col: "xA per 90"},
                    color_discrete_map={
                        "Double Threat": "#f59e0b",  # Amber/Orange
                        "Goal Threat": "#ef4444",    # Red
                        "Creative Hub": "#06b6d4",   # Cyan
                        "Rotation/Low": "#9ca3af"    # Gray
                    }
                )
                fig.add_vline(x=avg_xa, line_dash="dash", line_color="gray", annotation_text="Avg xA")
                fig.add_hline(y=avg_xg, line_dash="dash", line_color="gray", annotation_text="Avg xG")
                
                # Legend adjustment: "Down and left a bit"
                # Moving it to top-left (x=0, y=1.1) or strictly below?
                # Usually Plotly legends are right-aligned. Top-left horizontal is a good alternative.
                # Let's try x=0, y=1.1, orientation='h'
                fig.update_layout(
                    template="plotly_white", 
                    height=500,
                    legend=dict(
                        orientation='h', 
                        yanchor='bottom', y=1.02, 
                        xanchor='left', x=0  # Shifted left
                    ),
                    margin=dict(t=50) # Increase top margin for legend
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Legend Note
                st.caption(f"xP Column uses {con_ep_label} model.")
            else:
                st.info("No players match the filtered criteria")

            
    # Ownership Trends (always at the bottom, outside expander)
    render_ownership_trends(df)
