"""Analytics tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import (
    safe_numeric, style_df_with_injuries, round_df,
    classify_ownership_column, normalize_name,
    calculate_consensus_ep, get_consensus_label
)
from components.charts import create_cbit_chart, create_ownership_trends_chart

# Global color mapping for positions
POS_COLORS = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}


def format_with_rank(val, rank):
    """Format a value with a subscript rank."""
    if pd.isna(rank) or rank == "": return str(val)
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    rank_sub = str(int(rank)).translate(sub_map)
    return f"{val} ₍{rank_sub}₎"


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


def render_player_table(df: pd.DataFrame, horizon: int = 1):
    """Render the main player table with RRI differential columns and CBIT score (DEF/GKP only)."""
    ep_col = 'expected_points_display' if 'expected_points_display' in df.columns else 'expected_points'
    fpl_col = 'ep_next_display' if 'ep_next_display' in df.columns else 'ep_next'
    ml_col = 'ml_pred_display' if 'ml_pred_display' in df.columns else 'ml_pred'
    con_col = 'consensus_ep_display' if 'consensus_ep_display' in df.columns else 'consensus_ep'
    ac_col = 'avg_consensus_ep_display' if 'avg_consensus_ep_display' in df.columns else 'avg_consensus_ep'
    tp_col = 'total_points_display' if 'total_points_display' in df.columns else 'total_points'
    
    base_cols = ['web_name', 'team_name', 'position', 'now_cost', 'selected_by_percent',
                 ep_col, fpl_col, ml_col, con_col]
    
    if horizon > 1:
        base_cols.append(ac_col)
        
    base_cols.extend([tp_col, 'xnp', 'threat_momentum'])
    # Only show CBIT score if we have defenders/keepers (otherwise misleading)
    if 'cbit_score' in df.columns and df['position'].isin(['DEF', 'GKP']).any():
        base_cols.append('cbit_score')
    
    display_cols = [c for c in base_cols if c in df.columns]
    
    numeric_cols = ['now_cost', 'selected_by_percent', 'xnp', 'threat_momentum', 'cbit_score']
    # expected_points and ep_next are now strings in the display columns, or numeric in fallback
    if ep_col == 'expected_points': numeric_cols.append('expected_points')
    if fpl_col == 'ep_next': numeric_cols.append('ep_next')
    if ml_col == 'ml_pred': numeric_cols.append('ml_pred')
    if con_col == 'consensus_ep': numeric_cols.append('consensus_ep')

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    
    # Set CBIT to 0 for non-DEF/GKP to avoid NaN display issues
    if 'cbit_score' in df.columns:
        is_defensive = df['position'].isin(['DEF', 'GKP'])
        df.loc[~is_defensive, 'cbit_score'] = 0.0
    
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    con_ep_label = get_consensus_label(active_models, horizon)
    
    # Dynamic labels for specific models
    if horizon > 1:
        ep_label = f"Poisson EP ({horizon}GW)"
        fpl_ep_label = f"FPL EP x{horizon}"
        ml_ep_label = f"ML xP x{horizon}"
    else:
        ep_label = "Poisson EP"
        fpl_ep_label = "FPL EP"
        ml_ep_label = "ML xP"
    
    rename = {
        'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
        'now_cost': 'Price', 'selected_by_percent': 'Own%',
        ep_col: ep_label, fpl_col: fpl_ep_label, 
        ml_col: ml_ep_label, 
        con_col: con_ep_label,
        ac_col: f"Avg {con_ep_label}" if horizon > 1 else "Avg EP",
        tp_col: 'Total Points',
        'xnp': 'xNP', 'threat_momentum': 'Threat Momentum', 'cbit_score': 'CBIT',
    }
    
    # Show all players in scrollable table
    display_df = df[display_cols].copy()
    
    renamed = display_df.rename(columns=rename)
    # Ensure no duplicate column names (Styler requirement)
    renamed = renamed.loc[:, ~renamed.columns.duplicated()]
    
    st.markdown(f'<p class="section-title">Players ({len(df)} found)</p>', unsafe_allow_html=True)
    st.dataframe(style_df_with_injuries(renamed), hide_index=True, use_container_width=True, height=600)


def render_points_distribution(players_df: pd.DataFrame, con_ep_label: str):
    """Render xP distribution histogram by position."""
    st.markdown(f'<p class="section-title">{con_ep_label} Distribution by Position</p>', unsafe_allow_html=True)
    df = players_df.copy()
    # Use pre-calculated consensus_ep
    df['ep'] = safe_numeric(df.get('consensus_ep', df.get('ep_next', pd.Series([0]*len(df)))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 90]
    
    fig = go.Figure()
    for pos, color in POS_COLORS.items():
        pos_data = df[df['position'] == pos]['ep']
        if pos_data.empty:
            continue
        fig.add_trace(go.Histogram(
            x=pos_data, name=pos, marker_color=color,
            opacity=0.7, nbinsx=20
        ))
    
    fig.update_layout(
        height=320, barmode='overlay',
        template='plotly_white',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        xaxis=dict(title=con_ep_label, gridcolor='#e5e5ea'),
        yaxis=dict(title='Count', gridcolor='#e5e5ea'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=30, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='analytics_ep_distribution')


def render_value_by_position(players_df: pd.DataFrame, con_ep_label: str):
    """Render value (xP per million) box plot by position."""
    with st.expander("Metrics & Formulas"):
        st.markdown('''
        <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;margin-bottom:0.5rem;">
            <strong>Value Distribution Formula:</strong> <code>Value = xP ÷ Price</code> | Measures efficiency of point generation relative to cost.
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'''
        <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
            <strong>Value Analysis Formula:</strong> <code>xPPM ({con_ep_label} Per Million) = Cumulative {con_ep_label} ÷ Price</code> | <code>ROI/m = (xNP) ÷ Price</code>
        </div>
        ''', unsafe_allow_html=True)
    
    df = players_df.copy()
    # Use consensus_ep (Model xP) - the weighted blend of ML/Poisson/FPL
    df['ep'] = safe_numeric(df.get('consensus_ep', df.get('expected_points_poisson', pd.Series([0]*len(df)))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 200]
    df['value'] = df['ep'] / df['now_cost'].clip(lower=4)
    
    fig = go.Figure()
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_data = df[df['position'] == pos]
        if pos_data.empty:
            continue
        fig.add_trace(go.Box(
            y=pos_data['value'], name=pos,
            marker_color=POS_COLORS[pos],
            boxpoints='outliers',
            text=pos_data['web_name'],
            hovertemplate='<b>%{text}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        yaxis=dict(title='xP per £m', gridcolor='#e5e5ea'),
        showlegend=False,
        margin=dict(l=50, r=30, t=20, b=40)
    )
    # Value Leaderboard Filters
    vc1, vc2, vc3, vc4 = st.columns([1.5, 1, 1, 1])
    with vc1:
        v_search = st.text_input("Search player...", key="val_search", placeholder="Enter name")
    with vc2:
        v_sort = st.selectbox("Sort By", ["xP/m", "ROI/m", "Price"], key="val_sort")
    with vc3:
        v_max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="val_price")
    with vc4:
        v_min_mins = st.slider("Min Minutes", 0, 1000, 200, 90, key="val_mins")

    val_table = df.copy()
    val_table['now_cost'] = safe_numeric(val_table['now_cost'], 5)
    val_table['minutes'] = safe_numeric(val_table.get('minutes', 0))
    # Use consensus_ep (Model xP) - the weighted blend matching the main table
    if 'consensus_ep' in val_table.columns:
        val_table['ep_val'] = safe_numeric(val_table['consensus_ep'])
    else:
        val_table['ep_val'] = safe_numeric(val_table.get('expected_points_poisson', 0))
        
    val_table['eppm_val'] = val_table['ep_val'] / val_table['now_cost'].clip(lower=4)
    val_table['roi_val'] = safe_numeric(val_table.get('differential_gain', 0)) / val_table['now_cost'].clip(lower=4)

    # Apply Filters
    if v_search:
        query = normalize_name(v_search.lower().strip())
        val_table = val_table[val_table['web_name'].apply(lambda x: normalize_name(str(x).lower())).str.contains(query, na=False)]
    
    val_table = val_table[(val_table['now_cost'] <= v_max_price) & (val_table['minutes'] >= v_min_mins)]
    
    # Calculate Ranks on filtered set
    if not val_table.empty:
        val_table['eppm_rank'] = val_table['eppm_val'].rank(ascending=False, method='min').astype(int)
        val_table['roi_rank'] = val_table['roi_val'].rank(ascending=False, method='min').astype(int)
        
        val_table['eppm_display'] = val_table.apply(lambda r: format_with_rank(f"{r['eppm_val']:.2f}", r['eppm_rank']), axis=1)
        val_table['roi_display'] = val_table.apply(lambda r: format_with_rank(f"{r['roi_val']:.3f}", r['roi_rank']), axis=1)
    else:
        val_table['eppm_display'] = ""
        val_table['roi_display'] = ""

    val_display = val_table[['web_name', 'team_name', 'position', 'now_cost', 'eppm_display', 'roi_display']].copy()
    val_display.columns = ['Player', 'Team', 'Pos', 'Price', 'xP/m', 'ROI/m']
    
    # Sort mapping to handle display columns
    v_sort_map = {"xP/m": "eppm_val", "ROI/m": "roi_val", "Price": "now_cost"}
    v_asc = (v_sort == "Price")
    val_display = val_display.loc[val_table.sort_values(v_sort_map[v_sort], ascending=v_asc).index]
    
    st.dataframe(
        style_df_with_injuries(val_display, players_df),
        hide_index=True, use_container_width=True, height=400
    )

    # xP/m by position chart (moved here)
    st.markdown("### xP/m Distribution by Position")
    fig = go.Figure()
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_df = df[df['position'] == pos].copy()
        if pos_df.empty: continue
        # Use consensus_ep (Model xP) - the weighted blend matching the main table
        pos_ep = safe_numeric(pos_df.get('consensus_ep', pos_df.get('expected_points_poisson', 0)))
        pos_df['eppm_chart'] = pos_ep / safe_numeric(pos_df['now_cost'], 5).clip(lower=4)
        top_pos = pos_df.nlargest(10, 'eppm_chart')
        
        fig.add_trace(go.Bar(
            x=top_pos['web_name'], y=top_pos['eppm_chart'],
            name=pos, marker_color=POS_COLORS[pos],
            hovertemplate='<b>%{x}</b><br>xP/m: %{y:.2f}<br>Price: %{customdata[0]:.1f}m<extra></extra>',
            customdata=top_pos[['now_cost']].values
        ))
    fig.update_layout(
        height=300, template='plotly_white',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        xaxis=dict(gridcolor='#e5e5ea', tickangle=45),
        yaxis=dict(title='xP/m', gridcolor='#e5e5ea'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=30, t=40, b=80),
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True, key='analytics_eppm_chart_v2')


def render_cbit_analysis(players_df: pd.DataFrame):
    """Render CBIT (Clearances, Blocks, Interceptions, Tackles) analysis for all available players."""
    with st.expander("Metrics & Formulas"):
        st.markdown('''
        <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
            <strong>CBIT (Clearances, Blocks, Interceptions, Tackles) Formula:</strong> <code>Score = (AA90 × 0.4 + P(CBIT) × 5 + Floor × 0.2 + DTT × 0.1)</code> | AA90: Active Actions per 90m
        </div>
        ''', unsafe_allow_html=True)
    
    # Show chart
    fig = create_cbit_chart(players_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key='analytics_cbit_chart')
    
    # CBIT metrics table
    df = players_df.copy()
    cbit_cols = ['cbit_aa90', 'cbit_prob', 'cbit_floor', 'cbit_dtt', 'cbit_matchup', 'cbit_score', 'minutes']
    
    # Ensure all required columns exist and are numeric
    for col in cbit_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = safe_numeric(df[col])
    
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    
    # Filter for players who actually have CBIT data (AA90 > 0)
    df = df[df['cbit_aa90'] > 0]
    
    if not df.empty:
        # CBIT Filters
        st.markdown("---")
        cc1, cc2, cc3, cc4 = st.columns([1.5, 1, 1, 1])
        with cc1:
            c_search = st.text_input("Search player...", key="cbit_search", placeholder="Enter name")
        with cc2:
            c_sort = st.selectbox("Sort By", ["Score", "AA90", "P(CBIT)", "Price"], key="cbit_sort_v3")
        with cc3:
            c_max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="cbit_p_final")
        with cc4:
            c_min_mins = st.slider("Min Minutes", 0, 1000, 200, 90, key="cbit_m_final")

        # Apply Filters
        if c_search:
            query = normalize_name(c_search.lower().strip())
            df = df[df['web_name'].apply(lambda x: normalize_name(str(x).lower())).str.contains(query, na=False)]
        
        df = df[(df['now_cost'] <= c_max_price) & (df['minutes'] >= c_min_mins)]
        
        if not df.empty:
            # Calculate Ranks on filtered set
            df['score_rank'] = df['cbit_score'].rank(ascending=False, method='min').astype(int)
            df['aa90_rank'] = df['cbit_aa90'].rank(ascending=False, method='min').astype(int)
            df['prob_rank'] = df['cbit_prob'].rank(ascending=False, method='min').astype(int)
            
            # Correct display format to 2dp for points
            df['score_display'] = df.apply(lambda r: format_with_rank(f"{r['cbit_score']:.2f}", r['score_rank']), axis=1)
            df['aa90_display'] = df.apply(lambda r: format_with_rank(f"{r['cbit_aa90']:.2f}", r['aa90_rank']), axis=1)
            df['prob_display'] = df.apply(lambda r: format_with_rank(f"{r['cbit_prob']:.0%}", r['prob_rank']), axis=1)

            team_col = 'team_name' if 'team_name' in df.columns else 'team'
            display_df = df[['web_name', team_col, 'now_cost', 'aa90_display', 'prob_display', 'cbit_floor', 'cbit_dtt', 'score_display', 'minutes']].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'AA90', 'P(CBIT)', 'Floor', 'DTT', 'Score', 'minutes']
            
            # Sort mapping
            c_sort_mapping = {"Score": "cbit_score", "AA90": "cbit_aa90", "P(CBIT)": "cbit_prob", "Price": "now_cost"}
            c_asc = (c_sort == "Price")
            # We must use df to get indices for display_df
            sorted_index = df.sort_values(c_sort_mapping[c_sort], ascending=c_asc).index
            display_df = display_df.loc[sorted_index]

            display_df['Price'] = display_df['Price'].apply(lambda x: f"£{x:.1f}m")
            display_df['Floor'] = display_df['Floor'].apply(lambda x: f"{x:.1f}")
            display_df['DTT'] = display_df['DTT'].apply(lambda x: f"{x:+.1f}")
            
            # Remove minutes from display but used for filtering
            display_df = display_df.drop(columns=['minutes'])

            # Fixed height for preview (400)
            st.dataframe(style_df_with_injuries(display_df, players_df), hide_index=True, use_container_width=True, height=400, key='cbit_table_final')
        else:
            st.info("No players match the CBIT filter criteria")
    else:
        st.info("CBIT metrics unavailable - requires defensive action data")


def render_advanced_metrics(players_df: pd.DataFrame, con_ep_label: str):
    """Render Advanced Metrics section: EPPM, Threat Momentum, Matchup Quality, Engineered Differentials."""
    st.markdown('<p class="section-title">Advanced Metrics</p>', unsafe_allow_html=True)
    st.caption("xP/m, Threat Momentum & Matchup Quality — find hidden value using xG/xA data")
    
    df = players_df.copy()
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 200]  # Filter for players with meaningful minutes
    
    # Ensure columns exist
    for col in ['eppm', 'threat_momentum', 'matchup_quality', 'engineered_diff', 'threat_direction']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = safe_numeric(df[col])
    
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    # Use consensus_ep (Model xP) - the weighted blend of ML/Poisson/FPL
    df['model_xp'] = safe_numeric(df.get('consensus_ep', df.get('expected_points_poisson', pd.Series([0]*len(df)))))
    
    # ── Sub-tabs for different views ──
    am1, am2 = st.tabs(["Threat Momentum", "Engineered Differentials"])
    
    with am1:
        # Threat Momentum explanation
        with st.expander("Metrics & Formulas"):
            st.markdown('''
            <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
                <strong>Threat Momentum Formula:</strong> <code>momentum = rolling_xG + rolling_xA</code> | measures short-term attacking form
            </div>
            ''', unsafe_allow_html=True)
        # Threat Momentum: Searchable scatter plot
        st.markdown("**Threat Momentum (xG/xA Trend)**")
        st.caption("Search for a player to highlight on the chart and see detailed stats")
        
        # Search input
        am2_search = st.text_input("Search player", placeholder="Type name to highlight...", key="am2_player_search")
        
        # Prepare data for scatter
        scatter_df = df[df['threat_momentum'] > 0].copy()
        
        if 'matchup_quality' in scatter_df.columns and not scatter_df.empty:
            # Determine if we're searching
            is_searching = bool(am2_search and am2_search.strip())
            search_lower = am2_search.lower().strip() if is_searching else ""
            
            if is_searching:
                search_norm = normalize_name(search_lower)
                scatter_df['_name_norm'] = scatter_df['web_name'].apply(lambda x: normalize_name(str(x).lower()))
                scatter_df['is_searched'] = scatter_df['_name_norm'].str.contains(search_norm, na=False)
            else:
                scatter_df['is_searched'] = False
            
            fig = go.Figure()
            
            # Plot non-matched players (transparent when searching)
            for pos, color in POS_COLORS.items():
                pos_df = scatter_df[(scatter_df['position'] == pos) & (~scatter_df['is_searched'])]
                if pos_df.empty:
                    continue
                opacity = 0.15 if is_searching else 0.7
                fig.add_trace(go.Scatter(
                    x=pos_df['threat_momentum'], y=pos_df['matchup_quality'],
                    mode='markers', name=pos,
                    marker=dict(size=8, color=color, opacity=opacity),
                    text=pos_df['web_name'],
                    hovertemplate='<b>%{text}</b><br>Momentum: %{x:.2f}<br>Matchup: %{y:.2f}<extra></extra>'
                ))
            
            # Highlight matched players
            matched_df = scatter_df[scatter_df['is_searched']]
            if not matched_df.empty:
                for _, p in matched_df.iterrows():
                    pos_color = POS_COLORS.get(p['position'], '#ffffff')
                    fig.add_trace(go.Scatter(
                        x=[p['threat_momentum']], y=[p['matchup_quality']],
                        mode='markers+text', name=p['web_name'],
                        marker=dict(size=16, color=pos_color, symbol='diamond',
                                   line=dict(width=2, color='#ffffff')),
                        text=[p['web_name']], textposition='top center',
                        textfont=dict(color='#ffffff', size=12),
                        hovertemplate=f"<b>{p['web_name']}</b><br>Momentum: {p['threat_momentum']:.2f}<br>Matchup: {p['matchup_quality']:.2f}<extra></extra>",
                        showlegend=False
                    ))
            
            fig.update_layout(
                height=380, template='plotly_white',
                paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                font=dict(family='Inter, sans-serif', color='#86868b', size=11),
                xaxis=dict(title='Threat Momentum', gridcolor='#e5e5ea'),
                yaxis=dict(title='Matchup Quality', gridcolor='#e5e5ea'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=50, r=30, t=40, b=50)
            )
            
            # Add quadrant labels
            x_mid = scatter_df['threat_momentum'].median()
            y_mid = scatter_df['matchup_quality'].median()
            fig.add_annotation(x=scatter_df['threat_momentum'].quantile(0.9), y=scatter_df['matchup_quality'].quantile(0.9),
                              text="Sweet Spot", showarrow=False, font=dict(color='#22c55e', size=10))
            fig.add_annotation(x=scatter_df['threat_momentum'].quantile(0.1), y=scatter_df['matchup_quality'].quantile(0.9),
                              text="Good Matchup", showarrow=False, font=dict(color='#3b82f6', size=9))
            fig.add_annotation(x=scatter_df['threat_momentum'].quantile(0.9), y=scatter_df['matchup_quality'].quantile(0.1),
                              text="High Threat", showarrow=False, font=dict(color='#f59e0b', size=9))
            
            st.plotly_chart(fig, use_container_width=True, key='analytics_momentum_scatter')
            
            # Show detailed stats for matched player(s)
            if is_searching and not matched_df.empty:
                st.markdown("**Player Details**")
                for _, p in matched_df.iterrows():
                    pos_color = POS_COLORS.get(p['position'], '#888')
                    # Gather all available stats
                    xg = safe_numeric(pd.Series([p.get('us_xG', p.get('expected_goals', 0))])).iloc[0]
                    xa = safe_numeric(pd.Series([p.get('us_xA', p.get('expected_assists', 0))])).iloc[0]
                    goals = int(safe_numeric(pd.Series([p.get('goals_scored', 0)])).iloc[0])
                    assists = int(safe_numeric(pd.Series([p.get('assists', 0)])).iloc[0])
                    ep = safe_numeric(pd.Series([p.get('expected_points', 0)])).iloc[0]
                    eppm = safe_numeric(pd.Series([p.get('eppm', 0)])).iloc[0]
                    form = safe_numeric(pd.Series([p.get('form', 0)])).iloc[0]
                    own = safe_numeric(pd.Series([p.get('selected_by_percent', 0)])).iloc[0]
                    momentum = safe_numeric(pd.Series([p.get('threat_momentum', 0)])).iloc[0]
                    matchup = safe_numeric(pd.Series([p.get('matchup_quality', 0)])).iloc[0]
                    direction = safe_numeric(pd.Series([p.get('threat_direction', 0)])).iloc[0]
                    
                    # Over/underperformance
                    goal_overperf = goals - xg
                    assist_overperf = assists - xa
                    goal_label = f"+{goal_overperf:.2f}" if goal_overperf >= 0 else f"{goal_overperf:.2f}"
                    goal_color = '#22c55e' if goal_overperf > 0 else '#ef4444' if goal_overperf < 0 else '#888'
                    assist_label = f"+{assist_overperf:.2f}" if assist_overperf >= 0 else f"{assist_overperf:.2f}"
                    assist_color = '#22c55e' if assist_overperf > 0 else '#ef4444' if assist_overperf < 0 else '#888'
                    
                    direction_label = f"+{direction:.0%}" if direction >= 0 else f"{direction:.0%}"
                    direction_color = '#22c55e' if direction > 0 else '#ef4444' if direction < 0 else '#888'
                    
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;margin-bottom:0.5rem;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">'
                        f'<div><span style="color:{pos_color};font-weight:600;font-size:0.9rem;">{p["position"]}</span>'
                        f' <span style="color:#1d1d1f;font-weight:700;font-size:1.2rem;">{p["web_name"]}</span>'
                        f' <span style="color:#888;font-size:0.85rem;">{p.get("team_name", "")} | {p["now_cost"]:.1f}m</span></div>'
                        f'<div style="color:#888;font-size:0.8rem;">Own: {own:.1f}%</div></div>'
                        f'<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.75rem;text-align:center;">'
                        f'<div><div style="color:#888;font-size:0.7rem;">xP</div><div style="color:#1d1d1f;font-weight:600;">{ep:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">xP/m</div><div style="color:#1d1d1f;font-weight:600;">{eppm:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Form</div><div style="color:#1d1d1f;font-weight:600;">{form:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Direction</div><div style="color:{direction_color};font-weight:600;">{direction_label}</div></div>'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.75rem;text-align:center;margin-top:0.5rem;">'
                        f'<div><div style="color:#888;font-size:0.7rem;">Momentum</div><div style="color:#1d1d1f;font-weight:600;">{momentum:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Matchup</div><div style="color:#1d1d1f;font-weight:600;">{matchup:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Goals vs xG</div><div style="color:{goal_color};font-weight:600;">{goals} ({goal_label})</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Assists vs xA</div><div style="color:{assist_color};font-weight:600;">{assists} ({assist_label})</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("Matchup quality data unavailable")
    
    with am2:
        # Differential explanation
        with st.expander("Metrics & Formulas"):
            st.markdown('''
            <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
                <strong>Engineered Differentials Formula:</strong> <code>Score = xNP × (1 + momentum) × matchup_quality</code> | xNP: Expected Net Points
            </div>
            ''', unsafe_allow_html=True)
        # Differential Filters
        st.markdown("---")
        dc1, dc2, dc3, dc4 = st.columns([1.5, 1, 1, 1])
        with dc1:
            d_search = st.text_input("Search differential...", key="diff_search", placeholder="Enter name")
        with dc2:
            d_sort = st.selectbox("Sort By", ["Score", "xNP", "ROI/m", "Price"], key="diff_sort_v2")
        with dc3:
            d_max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="diff_price")
        with dc4:
            d_min_mins = st.slider("Min Minutes", 0, 1000, 200, 90, key="diff_mins")

        eng_sort = 'engineered_diff' if 'engineered_diff' in df.columns else 'differential_gain'
        
        # Base filter for differentials (<10% ownership)
        eng_table = df[df['selected_by_percent'] < 10].copy()
        
        # Apply Filters
        if d_search:
            query = normalize_name(d_search.lower().strip())
            eng_table = eng_table[eng_table['web_name'].apply(lambda x: normalize_name(str(x).lower())).str.contains(query, na=False)]
        
        eng_table = eng_table[(eng_table['now_cost'] <= d_max_price) & (eng_table['minutes'] >= d_min_mins)]
        
        if not eng_table.empty:
            eng_display_cols = ['web_name', 'team_name', 'position', 'now_cost', 'consensus_ep',
                                'differential_gain', 'diff_roi', 'eo_top10k', 'matchup_quality', 'selected_by_percent', eng_sort]
            eng_display_cols = [c for c in eng_display_cols if c in eng_table.columns]
            display_eng = eng_table[eng_display_cols].copy()
            eng_rename = {
                'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
                'now_cost': 'Price', 'consensus_ep': con_ep_label, 'differential_gain': 'xNP',
                'diff_roi': 'ROI/m', 'eo_top10k': 'EO10k%', 'matchup_quality': 'Matchup',
                'selected_by_percent': 'Own%', 'engineered_diff': 'Score',
            }
            display_eng = display_eng.rename(columns=eng_rename)
            
            # Sort
            d_sort_map = {"Score": "Score", "xNP": "xNP", "ROI/m": "ROI/m", "Price": "Price"}
            d_asc = (d_sort == "Price")
            display_eng = display_eng.sort_values(d_sort_map.get(d_sort, "Score"), ascending=d_asc)

            for c in ['Price', 'xP', 'xNP', 'ROI/m', 'EO10k%', 'Own%', 'Score']:
                if c in display_eng.columns:
                    display_eng[c] = display_eng[c].round(2)
            
            st.dataframe(style_df_with_injuries(display_eng), hide_index=True, use_container_width=True, height=400)
        else:
            st.info("No engineered differentials found")
        
        # Highlight top 3
        if len(eng_table) >= 3:
            st.markdown("**Top 3 Smart Buys**")
            top3_cols = st.columns(3)
            for i, (_, p) in enumerate(eng_table.head(3).iterrows()):
                with top3_cols[i]:
                    pos_color = POS_COLORS.get(p['position'], '#888')
                    diff_gain = p.get('differential_gain', 0)
                    verdict = p.get('diff_verdict', '')
                    profile = p.get('diff_profile', '')
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:0.7rem;">'
                        f'<div style="color:{pos_color};font-size:0.75rem;font-weight:600;">{p["position"]}</div>'
                        f'<div style="color:#1d1d1f;font-size:1rem;font-weight:600;">{p["web_name"]}</div>'
                        f'<div style="color:#888;font-size:0.8rem;">{p.get("team_name", "")} | {p["now_cost"]:.1f}m</div>'
                        f'<div style="margin-top:0.4rem;">'
                        f'<span style="color:#22c55e;">xNP {diff_gain:.2f}</span> | '
                        f'<span style="color:#f59e0b;">Own {p["selected_by_percent"]:.1f}%</span>'
                        f'{" | " + verdict if verdict else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )


def render_set_and_forget(players_df: pd.DataFrame, con_ep_label: str):
    """Render Set & Forget finder."""
    with st.expander("Metrics & Formulas"):
        st.markdown('''
        <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
            <strong>Set & Forget Formula:</strong> <code>Score = xP × 0.5 + Form × 0.2 + MinutesReliability × 3</code> | Measures consistency and output.
        </div>
        ''', unsafe_allow_html=True)
    
    df = players_df.copy()
    # Use consensus_ep (Model xP) - the weighted blend of ML/Poisson/FPL
    df['ep'] = safe_numeric(df.get('consensus_ep', df.get('expected_points_poisson', pd.Series([0]*len(df)))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df['form'] = safe_numeric(df.get('form', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    
    df['minutes_reliability'] = df['minutes'].clip(upper=2500) / 2500
    df['sf_score'] = df['ep'] * 0.5 + df['form'] * 0.2 + df['minutes_reliability'] * 10 * 0.3
    
    sf_cols = st.columns(4)
    
    for i, pos in enumerate(['GKP', 'DEF', 'MID', 'FWD']):
        with sf_cols[i]:
            st.markdown(f"**{pos}**")
            pos_df = df[df['position'] == pos].nlargest(3, 'sf_score')[['web_name', 'now_cost', 'sf_score']]
            pos_df.columns = ['Player', 'Price', 'S&F Score']
            pos_df['S&F Score'] = pos_df['S&F Score'].round(1)
            st.dataframe(style_df_with_injuries(pos_df), hide_index=True, use_container_width=True)


def render_expected_vs_actual(players_df: pd.DataFrame, con_ep_label: str):
    """Render Expected vs Actual performance analysis with filters."""
    with st.expander("Metrics & Formulas"):
        st.markdown('''
        <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;font-size:0.85rem;color:#1d1d1f;">
            <strong>Expected vs Actual:</strong> Compares points achieved (<code>Total Points</code>) against the <code>Comprehensive xPts</code> model (sum of all expected actions).
        </div>
        ''', unsafe_allow_html=True)
    
    df = players_df.copy()
    df['total_points'] = safe_numeric(df.get('total_points', pd.Series([0]*len(df))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 500]
    
    df['games_played'] = df['minutes'] / 90
    
    # ── Position-based point values ──
    GOAL_PTS = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
    CS_PTS = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
    GC_PENALTY = {'GKP': -0.5, 'DEF': -0.5, 'MID': 0, 'FWD': 0}  # per goal conceded (every 2)
    
    # ── xG/xA: prefer Understat, fallback to FPL ──
    xg_col = 'us_xG' if 'us_xG' in df.columns and df['us_xG'].sum() > 0 else 'expected_goals'
    xa_col = 'us_xA' if 'us_xA' in df.columns and df['us_xA'].sum() > 0 else 'expected_assists'
    for col in (xg_col, xa_col):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = safe_numeric(df[col])
    
    # ── Clean Sheet Rate (season average) ──
    # Use Poisson p_cs if available, else historical CS rate
    if 'poisson_p_cs' in df.columns:
        df['_p_cs'] = safe_numeric(df['poisson_p_cs']).clip(0, 0.7)
    else:
        df['clean_sheets'] = safe_numeric(df.get('clean_sheets', pd.Series([0]*len(df))))
        df['_p_cs'] = (df['clean_sheets'] / df['games_played'].clip(lower=1)).clip(0, 0.7)
    
    # ── CBIT Probability (season average, not next-match specific) ──
    # Use hit rate if available (season %), else position baseline
    if 'cbit_hit_rate' in df.columns:
        df['_p_cbit'] = safe_numeric(df['cbit_hit_rate']).clip(0, 1)
    elif 'cbit_prob' in df.columns:
        # cbit_prob is next-match, but better than nothing
        df['_p_cbit'] = safe_numeric(df['cbit_prob']).clip(0, 1) * 0.8  # Discount 20% for season average
    else:
        # Position baseline estimates
        df['_p_cbit'] = df['position'].map({'GKP': 0.30, 'DEF': 0.40, 'MID': 0.10, 'FWD': 0.05}).fillna(0.1)
    
    # ── Saves (GKP only) ──
    df['saves'] = safe_numeric(df.get('saves', pd.Series([0]*len(df))))
    
    # ── Goals Conceded ──
    df['goals_conceded'] = safe_numeric(df.get('goals_conceded', pd.Series([0]*len(df))))
    
    # ── Bonus Points Estimate ──
    # Higher xG+xA involvement correlates with bonus; ~0.8 bonus per xG+xA combo
    df['_xbonus'] = ((df[xg_col] + df[xa_col]) * 0.8).clip(0, df['games_played'] * 3)
    
    # ══════════════════════════════════════════════════════════════
    # COMPREHENSIVE EXPECTED POINTS FORMULA
    # ══════════════════════════════════════════════════════════════
    df['xpts_appearance'] = df['games_played'] * 2  # 2 pts per 60+ min appearance
    df['xpts_goals'] = df[xg_col] * df['position'].map(GOAL_PTS).fillna(4)
    df['xpts_assists'] = df[xa_col] * 3
    df['xpts_cs'] = df['games_played'] * df['_p_cs'] * df['position'].map(CS_PTS).fillna(0)
    df['xpts_cbit'] = df['games_played'] * df['_p_cbit'] * 2  # CBIT bonus = 2 pts
    df['xpts_saves'] = (df['saves'] / 3).clip(lower=0)  # 1 pt per 3 saves
    df['xpts_gc_penalty'] = df['goals_conceded'] * df['position'].map(GC_PENALTY).fillna(0)
    df['xpts_bonus'] = df['_xbonus']
    
    df['expected_total'] = (
        df['xpts_appearance'] +
        df['xpts_goals'] +
        df['xpts_assists'] +
        df['xpts_cs'] +
        df['xpts_cbit'] +
        df['xpts_saves'] +
        df['xpts_gc_penalty'] +
        df['xpts_bonus']
    ).round(1)
    
    df['diff'] = (df['total_points'] - df['expected_total']).round(1)

    # ── Filters ──
    ef1, ef2, ef3 = st.columns([1, 1, 2])
    with ef1:
        eva_pos = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="eva_pos")
    with ef2:
        all_teams = sorted(df['team_name'].dropna().unique().tolist()) if 'team_name' in df.columns else []
        eva_team = st.selectbox("Team", ['All'] + all_teams, key="eva_team")
    with ef3:
        eva_search = st.text_input("Search player", placeholder="Type name to highlight...", key="eva_search")

    if eva_pos != 'All':
        df = df[df['position'] == eva_pos]
    if eva_team != 'All' and 'team_name' in df.columns:
        df = df[df['team_name'] == eva_team]
    
    # Scatter chart: actual vs expected
    top_players = df.nlargest(60, 'total_points')
    if len(top_players) > 5:
        fig = go.Figure()
        
        # Diagonal line (x=y)
        max_val = max(top_players['total_points'].max(), top_players['expected_total'].max()) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='rgba(0,0,0,0.08)', dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
        
        for pos, color in POS_COLORS.items():
            pos_df = top_players[top_players['position'] == pos]
            if pos_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pos_df['expected_total'], y=pos_df['total_points'],
                mode='markers', name=pos,
                marker=dict(size=8, color=color, opacity=0.75,
                           line=dict(width=1, color='rgba(0,0,0,0.08)')),
                text=pos_df['web_name'],
                hovertemplate='<b>%{text}</b><br>Stats xP: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>'
            ))

        # Highlight searched player
        if eva_search and eva_search.strip():
            search_lower = eva_search.lower().strip()
            search_norm = normalize_name(search_lower)
            top_players['_name_norm'] = top_players['web_name'].apply(lambda x: normalize_name(str(x).lower()))
            matched = top_players[top_players['_name_norm'].str.contains(search_norm, na=False)]
            if not matched.empty:
                fig.add_trace(go.Scatter(
                    x=matched['expected_total'], y=matched['total_points'],
                    mode='markers+text', name='Search',
                    marker=dict(size=14, color='#ffffff', symbol='diamond',
                               line=dict(width=2, color='#ef4444')),
                    text=matched['web_name'], textposition='top center',
                    textfont=dict(color='#ffffff', size=11),
                    hovertemplate='<b>%{text}</b><br>Stats xP: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>'
                ))
        
        fig.update_layout(
            height=380, template='plotly_white',
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(title='Stats-based xP (Action Quality Sum)', gridcolor='#e5e5ea'),
            yaxis=dict(title='Actual Total Points', gridcolor='#e5e5ea'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=30, t=30, b=50)
        )
        fig.add_annotation(x=max_val*0.15, y=max_val*0.85, text="Overperforming",
                          showarrow=False, font=dict(color='#22c55e', size=10))
        fig.add_annotation(x=max_val*0.85, y=max_val*0.15, text="Underperforming",
                          showarrow=False, font=dict(color='#ef4444', size=10))
        st.plotly_chart(fig, use_container_width=True, key='analytics_expected_vs_actual')
    
    # Add threat momentum to the data
    if 'threat_momentum' not in df.columns:
        df['threat_momentum'] = 0.0
    df['threat_momentum'] = safe_numeric(df['threat_momentum'])
    if 'threat_direction' not in df.columns:
        df['threat_direction'] = 0.0
    df['threat_direction'] = safe_numeric(df['threat_direction'])
    
    xva1, xva2 = st.columns(2)
    
    with xva1:
        st.markdown("**Underperformers** (unlucky, may bounce back)")
        st.caption("Comprehensive xPts includes CS, CBIT, saves, bonus — positive threat = heating up")
        under = df.nsmallest(8, 'diff')[['web_name', 'position', 'total_points', 'expected_total', 'diff', 'threat_momentum']].copy()
        under.columns = ['Player', 'Pos', 'Actual', 'xPts', 'Diff', 'Threat']
        under['xPts'] = under['xPts'].round(0).astype(int)
        under['Diff'] = under['Diff'].round(0).astype(int)
        under['Threat'] = under['Threat'].round(2)
        st.dataframe(style_df_with_injuries(under), hide_index=True, use_container_width=True)
    
    with xva2:
        st.markdown("**Overperformers** (may regress)")
        st.caption("Players outperforming xPts model — negative threat = cooling off")
        over = df.nlargest(8, 'diff')[['web_name', 'position', 'total_points', 'expected_total', 'diff', 'threat_momentum']].copy()
        over.columns = ['Player', 'Pos', 'Actual', 'xPts', 'Diff', 'Threat']
        over['xPts'] = over['xPts'].round(0).astype(int)
        over['Diff'] = over['Diff'].round(0).astype(int)
        over['Threat'] = over['Threat'].round(2)
        st.dataframe(style_df_with_injuries(over), hide_index=True, use_container_width=True)


def render_ownership_trends(players_df: pd.DataFrame):
    """Render ownership trends chart showing transfers in/out."""
    st.markdown('<p class="section-title">Ownership Trends</p>', unsafe_allow_html=True)
    st.caption("Most transferred players this gameweek — green = in, red = out")
    
    fig = create_ownership_trends_chart(players_df, limit=20)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key='analytics_ownership_trends')
    else:
        st.info("Transfer data unavailable")
