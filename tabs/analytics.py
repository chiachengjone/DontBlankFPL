"""Analytics tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import (
    safe_numeric, style_df_with_injuries, round_df,
    classify_ownership_column, normalize_name,
)
from components.charts import create_cbit_chart, create_ownership_trends_chart


def render_analytics_tab(processor, players_df: pd.DataFrame):
    """Analytics tab - player discovery and advanced metrics."""
    
    # Filters - use session state defaults
    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1])
    
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
    with f4:
        sort_col = st.selectbox(
            "Sort By",
            ['Expected Points', 'EPPM', 'Threat', 'Diff Gain (RRI)', 'Diff ROI/m', 'CBIT', 'Price', 'Ownership'],
            key="analytics_sort",
        )
    with f5:
        tier_filter = st.selectbox(
            "Ownership Tier",
            ['All', 'Template', 'Popular', 'Enabler', 'Differential'],
            key="analytics_tier",
        )
    
    search = st.text_input("Search player...", placeholder="Enter name", key="analytics_search")
    
    # Prepare data - use expected_points from fpl_api (already enriched with Understat)
    df = players_df.copy()
    # EP comes from advanced calculation in fpl_api; fall back to ep_next if missing
    if 'expected_points' not in df.columns or df['expected_points'].isna().all():
        df['expected_points'] = safe_numeric(df.get('ep_next', pd.Series([2.0]*len(df))))
    else:
        df['expected_points'] = safe_numeric(df['expected_points'])
    
    # Use pre-calculated differential metrics if available
    if 'differential_gain' not in df.columns:
        eo = safe_numeric(df['selected_by_percent'], 5).clip(lower=0.1)
        eo_frac = eo / 100.0
        import numpy as np
        eo_10k = (1.5 * np.power(eo_frac, 1.4) + 0.01).clip(0.01, 0.99)
        df['eo_top10k'] = (eo_10k * 100).round(1)
        df['differential_gain'] = (df['expected_points'] * (1 - eo_10k)).round(2)
        df['diff_roi'] = (df['differential_gain'] / safe_numeric(df['now_cost'], 5).clip(lower=4)).round(3)
    if 'differential_score' not in df.columns:
        df['differential_score'] = df['differential_gain']
    
    df['cbit_propensity'] = safe_numeric(df.get('cbit_propensity', pd.Series([0]*len(df))))
    if df['cbit_propensity'].sum() == 0 and 'clean_sheets' in df.columns:
        df.loc[df['position'] == 'DEF', 'cbit_propensity'] = safe_numeric(df['clean_sheets']) / 10
    
    df['xg_per_pound'] = safe_numeric(df.get('xg_per_pound', df['expected_points'] / safe_numeric(df['now_cost'], 5).clip(lower=4)))
    
    # Apply filters
    if pos_filter != 'All':
        df = df[df['position'] == pos_filter]
    
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['minutes'] = safe_numeric(df['minutes'])
    df = df[df['now_cost'] <= max_price]
    df = df[df['minutes'] >= min_mins]
    
    if search:
        query_norm = normalize_name(search.lower().strip())
        df['_name_norm'] = df['web_name'].apply(lambda x: normalize_name(str(x).lower()))
        df = df[df['_name_norm'].str.contains(query_norm, na=False)]
        df = df.drop(columns=['_name_norm'])
    
    # Sort
    sort_map = {
        'Expected Points': 'expected_points',
        'EPPM': 'eppm',
        'Threat': 'threat_momentum',
        'Diff Gain (RRI)': 'differential_gain',
        'Diff ROI/m': 'diff_roi',
        'Differential': 'differential_score',
        'CBIT': 'cbit_propensity',
        'Price': 'now_cost',
        'Ownership': 'selected_by_percent',
    }
    sort_by = sort_map.get(sort_col, 'expected_points')
    if sort_by in df.columns:
        df[sort_by] = safe_numeric(df[sort_by])
        df = df.sort_values(sort_by, ascending=(sort_col == 'Price'))
    
    # Add ownership tier
    df['ownership_tier'] = classify_ownership_column(df)
    
    # Apply tier filter
    if tier_filter != 'All':
        df = df[df['ownership_tier'] == tier_filter]
    
    # Display player table
    render_player_table(df)
    
    # Points distribution
    render_points_distribution(players_df)
    
    # Value by position
    render_value_by_position(players_df)
    
    # CBIT analysis chart
    render_cbit_analysis(players_df)
    
    # Advanced Metrics (Threat Momentum, EPPM, Matchup Quality)
    render_advanced_metrics(players_df)
    
    # Set & Forget finder
    render_set_and_forget(players_df)
    
    # Expected vs Actual
    render_expected_vs_actual(players_df)
    
    # Ownership Trends
    render_ownership_trends(players_df)


def render_player_table(df: pd.DataFrame):
    """Render the main player table with RRI differential columns and CBIT score (DEF/GKP only)."""
    base_cols = ['web_name', 'team_name', 'position', 'now_cost', 'selected_by_percent',
                 'expected_points', 'ep_next', 'xnp', 'diff_roi', 'eppm', 'threat_momentum']
    # Only show CBIT score if we have defenders/keepers (otherwise misleading)
    if 'cbit_score' in df.columns and df['position'].isin(['DEF', 'GKP']).any():
        base_cols.append('cbit_score')
    
    display_cols = [c for c in base_cols if c in df.columns]
    
    numeric_cols = ['now_cost', 'selected_by_percent', 'expected_points', 'ep_next',
                   'xnp', 'diff_roi', 'eppm', 'threat_momentum', 'cbit_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    
    # Set CBIT to 0 for non-DEF/GKP to avoid NaN display issues
    if 'cbit_score' in df.columns:
        is_defensive = df['position'].isin(['DEF', 'GKP'])
        df.loc[~is_defensive, 'cbit_score'] = 0.0
    
    rename = {
        'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
        'now_cost': 'Price', 'selected_by_percent': 'Own%',
        'expected_points': 'EP', 'ep_next': 'FPL EP', 'xnp': 'xNP',
        'diff_roi': 'ROI/m', 'eppm': 'EPPM', 'threat_momentum': 'Threat',
        'cbit_score': 'CBIT',
    }
    
    # Show all players in scrollable table
    display_df = df[display_cols].copy()
    
    renamed = display_df.rename(columns=rename)
    # Ensure no duplicate column names (Styler requirement)
    renamed = renamed.loc[:, ~renamed.columns.duplicated()]
    
    st.markdown(f'<p class="section-title">Players ({len(df)} found)</p>', unsafe_allow_html=True)
    st.dataframe(style_df_with_injuries(renamed), hide_index=True, use_container_width=True, height=600)


def render_points_distribution(players_df: pd.DataFrame):
    """Render EP distribution histogram by position."""
    st.markdown('<p class="section-title">EP Distribution by Position</p>', unsafe_allow_html=True)
    st.caption("How expected points are spread across each position")
    
    df = players_df.copy()
    # Use expected_points (advanced EP) if available, else fall back to ep_next
    df['ep'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([0]*len(df)))))
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 90]
    
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    fig = go.Figure()
    for pos, color in pos_colors.items():
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
        xaxis=dict(title='Expected Points', gridcolor='#e5e5ea'),
        yaxis=dict(title='Count', gridcolor='#e5e5ea'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=30, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='analytics_ep_distribution')


def render_value_by_position(players_df: pd.DataFrame):
    """Render value (EP per million) box plot by position."""
    st.markdown('<p class="section-title">Value Distribution (EP / Price)</p>', unsafe_allow_html=True)
    st.caption("Box plot showing EP per million by position — find the best bang for your buck")
    
    df = players_df.copy()
    # Use expected_points (advanced EP) if available
    df['ep'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([0]*len(df)))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    df = df[df['minutes'] > 200]
    df['value'] = df['ep'] / df['now_cost'].clip(lower=4)
    
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    fig = go.Figure()
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_data = df[df['position'] == pos]
        if pos_data.empty:
            continue
        fig.add_trace(go.Box(
            y=pos_data['value'], name=pos,
            marker_color=pos_colors[pos],
            boxpoints='outliers',
            text=pos_data['web_name'],
            hovertemplate='<b>%{text}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        yaxis=dict(title='EP per £m', gridcolor='#e5e5ea'),
        showlegend=False,
        margin=dict(l=50, r=30, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, key='analytics_value_boxplot')


def render_cbit_analysis(players_df: pd.DataFrame):
    """Render CBIT (Clearances, Blocks, Interceptions, Tackles) analysis for DEF/GKP."""
    cbit_hdr_cols = st.columns([3, 1])
    with cbit_hdr_cols[0]:
        st.markdown('<p class="section-title">CBIT Analysis (Defenders)</p>', unsafe_allow_html=True)
        st.caption("2025/26 DefCon scoring: DEF needs 10+ actions, GKP/MID/FWD need 12+ for +2 pts bonus")
    with cbit_hdr_cols[1]:
        show_all_cbit = st.button("Show All", key="show_all_cbit", help="Show all defenders sorted by CBIT score")
    
    # Show chart
    fig = create_cbit_chart(players_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key='analytics_cbit_chart')
    
    # Show CBIT metrics table for top defenders
    df = players_df[players_df['position'].isin(['DEF', 'GKP'])].copy()
    
    cbit_cols = ['cbit_aa90', 'cbit_prob', 'cbit_floor', 'cbit_dtt', 'cbit_matchup', 'cbit_score']
    if any(c in df.columns for c in cbit_cols):
        for col in cbit_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = safe_numeric(df[col])
        
        df['now_cost'] = safe_numeric(df['now_cost'], 5)
        
        if show_all_cbit or st.session_state.get('cbit_show_all', False):
            st.session_state['cbit_show_all'] = True
            df = df.sort_values('cbit_score' if 'cbit_score' in df.columns else 'cbit_aa90', ascending=False)
        else:
            df = df.nlargest(20, 'cbit_score' if 'cbit_score' in df.columns else 'cbit_aa90')
        
        if not df.empty:
            team_col = 'team_name' if 'team_name' in df.columns else 'team'
            display_df = df[['web_name', team_col, 'now_cost', 'cbit_aa90', 'cbit_prob', 'cbit_floor', 'cbit_dtt', 'cbit_score']].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'AA90', 'P(CBIT)', 'Floor', 'DTT', 'Score']
            display_df['Price'] = display_df['Price'].apply(lambda x: f"£{x:.1f}m")
            display_df['P(CBIT)'] = display_df['P(CBIT)'].apply(lambda x: f"{x:.0%}")
            display_df['Floor'] = display_df['Floor'].apply(lambda x: f"{x:.1f}")
            display_df['DTT'] = display_df['DTT'].apply(lambda x: f"{x:+.1f}")
            
            table_height = 600 if st.session_state.get('cbit_show_all', False) else min(400, len(display_df) * 40 + 40)
            st.dataframe(style_df_with_injuries(display_df, players_df), hide_index=True, use_container_width=True, height=table_height, key='cbit_table')
    else:
        st.info("CBIT metrics unavailable - requires defensive action data")


def render_advanced_metrics(players_df: pd.DataFrame):
    """Render Advanced Metrics section: EPPM, Threat Momentum, Matchup Quality, Engineered Differentials."""
    st.markdown('<p class="section-title">Advanced Metrics</p>', unsafe_allow_html=True)
    st.caption("EPPM, Threat Momentum & Matchup Quality — find hidden value using xG/xA data")
    
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
    df['expected_points'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([0]*len(df)))))
    
    # ── Sub-tabs for different views ──
    am1, am2, am3 = st.tabs(["EPPM Leaders", "Threat Momentum", "Engineered Differentials"])
    
    with am1:
        # EPPM: Effective Points Per Million
        hdr_cols = st.columns([3, 1])
        with hdr_cols[0]:
            st.markdown("**Top EPPM (Value Picks)**")
            st.caption("Highest expected points per million — best bang for your buck")
        with hdr_cols[1]:
            show_all_eppm = st.button("Show All", key="show_all_eppm", help="Show all players sorted by EPPM")
        
        if show_all_eppm or st.session_state.get('eppm_show_all', False):
            st.session_state['eppm_show_all'] = True
            eppm_df = df.sort_values('eppm', ascending=False)[['web_name', 'team_name', 'position', 'now_cost', 'expected_points', 'eppm', 'selected_by_percent']].copy()
        else:
            eppm_df = df.nlargest(20, 'eppm')[['web_name', 'team_name', 'position', 'now_cost', 'expected_points', 'eppm', 'selected_by_percent']].copy()
        
        eppm_df.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'EPPM', 'Own%']
        for c in ['Price', 'EP', 'EPPM', 'Own%']:
            eppm_df[c] = eppm_df[c].round(2)
        
        table_height = 600 if st.session_state.get('eppm_show_all', False) else min(400, len(eppm_df) * 40 + 40)
        st.dataframe(style_df_with_injuries(eppm_df), hide_index=True, use_container_width=True, height=table_height)
        
        # EPPM by position chart
        pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
        fig = go.Figure()
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = df[df['position'] == pos].nlargest(10, 'eppm')
            if pos_df.empty:
                continue
            fig.add_trace(go.Bar(
                x=pos_df['web_name'], y=pos_df['eppm'],
                name=pos, marker_color=pos_colors[pos],
                hovertemplate='<b>%{x}</b><br>EPPM: %{y:.2f}<br>Price: %{customdata[0]:.1f}m<extra></extra>',
                customdata=pos_df[['now_cost']].values
            ))
        fig.update_layout(
            height=300, template='plotly_white',
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(gridcolor='#e5e5ea', tickangle=45),
            yaxis=dict(title='EPPM', gridcolor='#e5e5ea'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=50, r=30, t=40, b=80),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True, key='analytics_eppm_chart')
    
    with am2:
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
            for pos, color in pos_colors.items():
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
                    pos_color = pos_colors.get(p['position'], '#ffffff')
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
                    pos_color = pos_colors.get(p['position'], '#888')
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
                    goal_label = f"+{goal_overperf:.1f}" if goal_overperf >= 0 else f"{goal_overperf:.1f}"
                    goal_color = '#22c55e' if goal_overperf > 0 else '#ef4444' if goal_overperf < 0 else '#888'
                    assist_label = f"+{assist_overperf:.1f}" if assist_overperf >= 0 else f"{assist_overperf:.1f}"
                    assist_color = '#22c55e' if assist_overperf > 0 else '#ef4444' if assist_overperf < 0 else '#888'
                    
                    direction_label = f"+{direction:.0%}" if direction >= 0 else f"{direction:.0%}"
                    direction_color = '#22c55e' if direction > 0 else '#ef4444' if direction < 0 else '#888'
                    
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;margin-bottom:0.5rem;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">'
                        f'<div><span style="color:{pos_color};font-weight:600;font-size:0.9rem;">{p["position"]}</span>'
                        f' <span style="color:#fff;font-weight:700;font-size:1.2rem;">{p["web_name"]}</span>'
                        f' <span style="color:#888;font-size:0.85rem;">{p.get("team_name", "")} | {p["now_cost"]:.1f}m</span></div>'
                        f'<div style="color:#888;font-size:0.8rem;">Own: {own:.1f}%</div></div>'
                        f'<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.75rem;text-align:center;">'
                        f'<div><div style="color:#888;font-size:0.7rem;">EP</div><div style="color:#1d1d1f;font-weight:600;">{ep:.1f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">EPPM</div><div style="color:#1d1d1f;font-weight:600;">{eppm:.2f}</div></div>'
                        f'<div><div style="color:#888;font-size:0.7rem;">Form</div><div style="color:#1d1d1f;font-weight:600;">{form:.1f}</div></div>'
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
    
    with am3:
        # Engineered Differentials: RRI-based smart buys
        diff_hdr_cols = st.columns([3, 1])
        with diff_hdr_cols[0]:
            st.markdown("**Engineered Differentials (RRI)**")
            st.caption("Low ownership (<10%) + High net-points gain + Positive momentum — Smart Buys")
        with diff_hdr_cols[1]:
            show_all_diff = st.button("Show All", key="show_all_diff", help="Show all differentials")
        
        eng_sort = 'engineered_diff' if 'engineered_diff' in df.columns else 'differential_gain'
        
        if show_all_diff or st.session_state.get('diff_show_all', False):
            st.session_state['diff_show_all'] = True
            eng_df = df[df['selected_by_percent'] < 10].sort_values(eng_sort, ascending=False)
        else:
            eng_df = df[df['selected_by_percent'] < 10].nlargest(20, eng_sort)
        if not eng_df.empty:
            eng_display_cols = ['web_name', 'team_name', 'position', 'now_cost', 'expected_points',
                                'differential_gain', 'diff_roi', 'eo_top10k', 'diff_profile', 'selected_by_percent', eng_sort]
            eng_display_cols = [c for c in eng_display_cols if c in eng_df.columns]
            display_eng = eng_df[eng_display_cols].copy()
            eng_rename = {
                'web_name': 'Player', 'team_name': 'Team', 'position': 'Pos',
                'now_cost': 'Price', 'expected_points': 'EP', 'differential_gain': 'xNP',
                'diff_roi': 'ROI/m', 'eo_top10k': 'EO10k%', 'diff_profile': 'Profile',
                'selected_by_percent': 'Own%', 'engineered_diff': 'Score',
            }
            display_eng = display_eng.rename(columns=eng_rename)
            for c in ['Price', 'EP', 'xNP', 'ROI/m', 'EO10k%', 'Own%', 'Score']:
                if c in display_eng.columns:
                    display_eng[c] = display_eng[c].round(2)
            
            table_height = 600 if st.session_state.get('diff_show_all', False) else min(400, len(display_eng) * 40 + 40)
            st.dataframe(style_df_with_injuries(display_eng), hide_index=True, use_container_width=True, height=table_height)
        else:
            st.info("No engineered differentials found")
        
        # Highlight top 3
        if len(eng_df) >= 3:
            st.markdown("**Top 3 Smart Buys**")
            top3_cols = st.columns(3)
            for i, (_, p) in enumerate(eng_df.head(3).iterrows()):
                with top3_cols[i]:
                    pos_color = pos_colors.get(p['position'], '#888')
                    diff_gain = p.get('differential_gain', 0)
                    verdict = p.get('diff_verdict', '')
                    profile = p.get('diff_profile', '')
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:0.7rem;">'
                        f'<div style="color:{pos_color};font-size:0.75rem;font-weight:600;">{p["position"]}</div>'
                        f'<div style="color:#fff;font-size:1rem;font-weight:600;">{p["web_name"]}</div>'
                        f'<div style="color:#888;font-size:0.8rem;">{p.get("team_name", "")} | {p["now_cost"]:.1f}m</div>'
                        f'<div style="margin-top:0.4rem;">'
                        f'<span style="color:#22c55e;">xNP {diff_gain:.2f}</span> | '
                        f'<span style="color:#f59e0b;">Own {p["selected_by_percent"]:.1f}%</span>'
                        f'{" | " + verdict if verdict else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )


def render_set_and_forget(players_df: pd.DataFrame):
    """Render Set & Forget finder."""
    st.markdown('<p class="section-title">Set & Forget Picks</p>', unsafe_allow_html=True)
    st.caption("Players with high EP, good fixture run, and consistent minutes - minimal rotation needed")
    
    df = players_df.copy()
    # Use expected_points (advanced EP) for Set & Forget
    df['ep'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([0]*len(df)))))
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


def render_expected_vs_actual(players_df: pd.DataFrame):
    """Render Expected vs Actual performance analysis with filters."""
    st.markdown('<p class="section-title">Expected vs Actual (Over/Under Performers)</p>', unsafe_allow_html=True)
    st.caption("Comprehensive xPts model: xG, xA, CS probability, CBIT, saves, goals conceded — find undervalued players")
    
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
        pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
        fig = go.Figure()
        
        # Diagonal line (x=y)
        max_val = max(top_players['total_points'].max(), top_players['expected_total'].max()) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='rgba(0,0,0,0.08)', dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
        
        for pos, color in pos_colors.items():
            pos_df = top_players[top_players['position'] == pos]
            if pos_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pos_df['expected_total'], y=pos_df['total_points'],
                mode='markers', name=pos,
                marker=dict(size=8, color=color, opacity=0.75,
                           line=dict(width=1, color='rgba(0,0,0,0.08)')),
                text=pos_df['web_name'],
                hovertemplate='<b>%{text}</b><br>Expected: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>'
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
                    hovertemplate='<b>%{text}</b><br>Expected: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>'
                ))
        
        fig.update_layout(
            height=380, template='plotly_white',
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(title='Expected Total Points', gridcolor='#e5e5ea'),
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
