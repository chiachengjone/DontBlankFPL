"""Optimization tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import safe_numeric, style_df_with_injuries, round_df
from optimizer import MAX_PLAYERS_PER_TEAM


def render_optimization_tab(processor, players_df: pd.DataFrame, fetcher):
    """Optimization tab - Transfer recommendations based on team."""
    
    st.markdown('<p class="section-title">Transfer Optimizer</p>', unsafe_allow_html=True)
    
    # Configuration
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        team_id = st.number_input("Your Team ID", min_value=1, max_value=10000000, value=1, key="opt_team_id")
    with c2:
        weeks_ahead = st.slider("Planning Horizon", min_value=3, max_value=10, value=5, key="opt_horizon")
    with c3:
        from fpl_api import MAX_FREE_TRANSFERS
        free_transfers = st.slider("Free Transfers", min_value=1, max_value=MAX_FREE_TRANSFERS, value=1, key="opt_fts")
    with c4:
        strategy = st.selectbox("Strategy", ['Balanced', 'Maximum Points', 'Differential', 'Value'], key="opt_strategy")
    
    bank = st.number_input("Bank (remaining budget)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, key="opt_bank")
    
    if st.button("Analyze Transfers", type="primary", use_container_width=True):
        with st.spinner("Analyzing your squad..."):
            try:
                analyze_transfers(processor, players_df, fetcher, team_id, weeks_ahead, free_transfers, strategy, bank)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter your Team ID and click 'Analyze Transfers' to get personalized recommendations.")


def analyze_transfers(processor, players_df, fetcher, team_id, weeks_ahead, free_transfers, strategy, bank):
    """Main transfer analysis logic."""
    
    featured_df = players_df.copy()
    
    # Get user's current team
    try:
        gw = fetcher.get_current_gameweek()
        picks_data = fetcher.get_team_picks(team_id, gw)
        if isinstance(picks_data, dict) and 'picks' in picks_data:
            your_squad = [p['element'] for p in picks_data['picks']]
        elif isinstance(picks_data, list):
            your_squad = [p.get('element', p) for p in picks_data]
        else:
            your_squad = []
        has_real_team = len(your_squad) > 0
    except Exception as e:
        your_squad = []
        has_real_team = False
        st.warning(f"Could not fetch team data: {e}. Showing general recommendations.")
    
    # Prepare EP column
    if 'expected_points' in featured_df.columns:
        ep_col = 'expected_points'
    elif 'ep_next' in featured_df.columns:
        ep_col = 'ep_next'
    else:
        featured_df['ep_next'] = 2.0
        ep_col = 'ep_next'
    
    featured_df[ep_col] = safe_numeric(featured_df[ep_col])
    featured_df['selected_by_percent'] = safe_numeric(featured_df['selected_by_percent'])
    featured_df['now_cost'] = safe_numeric(featured_df['now_cost'], 5)
    featured_df['minutes'] = safe_numeric(featured_df.get('minutes', pd.Series([0]*len(featured_df))))
    featured_df['form'] = safe_numeric(featured_df.get('form', pd.Series([0]*len(featured_df))))
    featured_df['total_points'] = safe_numeric(featured_df.get('total_points', pd.Series([0]*len(featured_df))))
    
    # Calculate transfer scores
    calculate_transfer_scores(featured_df, ep_col, strategy)
    
    st.markdown('<p class="section-title">Transfer Recommendations</p>', unsafe_allow_html=True)
    st.caption("Score 0-10: higher = better player to bring IN")
    
    if has_real_team and your_squad:
        your_squad_set = set(your_squad)
        current_squad_df = featured_df[featured_df['id'].isin(your_squad_set)].copy()
        available_df = featured_df[~featured_df['id'].isin(your_squad_set)].copy()
        
        # Team value tracker
        render_team_value_tracker(current_squad_df, bank)
        
        # Budget breakdown
        render_budget_breakdown(current_squad_df, ep_col)
        
        # Transfer recommendations
        render_transfer_recommendations(current_squad_df, available_df, ep_col)
        
        # AI transfer plan
        render_ai_transfer_plan(current_squad_df, available_df, ep_col, free_transfers, bank)
    else:
        available_df = featured_df.copy()
        current_squad_df = pd.DataFrame()
    
    # Position recommendations
    render_position_recommendations(available_df, ep_col)
    
    # Top 10 overall
    render_top_picks(available_df, ep_col)
    
    # Points projection
    if has_real_team and not current_squad_df.empty:
        render_points_projection(current_squad_df, available_df, ep_col)


def calculate_transfer_scores(df, ep_col, strategy):
    """Calculate transfer scores based on strategy."""
    # Normalize EP
    ep_min = df[ep_col].min()
    ep_max = df[ep_col].max()
    ep_range = max(ep_max - ep_min, 0.1)
    df['ep_norm'] = (df[ep_col] - ep_min) / ep_range * 10
    
    # Normalize form
    form_max = df['form'].max()
    df['form_norm'] = (df['form'] / max(form_max, 0.1)) * 10
    
    # Normalize value
    df['value_raw'] = df[ep_col] / df['now_cost'].clip(lower=4)
    v_max = df['value_raw'].max()
    df['value_norm'] = (df['value_raw'] / max(v_max, 0.1)) * 10
    
    # Minutes reliability
    df['mins_norm'] = (df['minutes'].clip(upper=2000) / 2000) * 10
    
    # Differential
    df['diff_norm'] = (100 - df['selected_by_percent'].clip(upper=100)) / 10
    
    # Calculate based on strategy
    if strategy == 'Maximum Points':
        df['transfer_score'] = df['ep_norm'] * 0.55 + df['form_norm'] * 0.25 + df['mins_norm'] * 0.20
    elif strategy == 'Differential':
        df['transfer_score'] = df['ep_norm'] * 0.35 + df['diff_norm'] * 0.35 + df['form_norm'] * 0.15 + df['mins_norm'] * 0.15
    elif strategy == 'Value':
        df['transfer_score'] = df['value_norm'] * 0.45 + df['ep_norm'] * 0.30 + df['mins_norm'] * 0.25
    else:  # Balanced
        df['transfer_score'] = df['ep_norm'] * 0.35 + df['form_norm'] * 0.20 + df['value_norm'] * 0.20 + df['mins_norm'] * 0.15 + df['diff_norm'] * 0.10
    
    # Scale to 0-10
    ts_min = df['transfer_score'].min()
    ts_max = df['transfer_score'].max()
    ts_range = max(ts_max - ts_min, 0.1)
    df['transfer_score'] = ((df['transfer_score'] - ts_min) / ts_range * 10).round(1)


def render_team_value_tracker(current_squad_df, bank):
    """Render team value metrics."""
    squad_value = current_squad_df['now_cost'].sum() if not current_squad_df.empty else 0
    total_budget = squad_value + bank
    
    tv1, tv2, tv3, tv4 = st.columns(4)
    with tv1:
        st.metric("Squad Value", f"{squad_value:.1f}m")
    with tv2:
        st.metric("Bank", f"{bank:.1f}m")
    with tv3:
        st.metric("Total Budget", f"{total_budget:.1f}m")
    with tv4:
        profit = total_budget - 100.0
        st.metric("Profit/Loss", f"{profit:+.1f}m", delta=f"{profit:+.1f}m")


def render_budget_breakdown(current_squad_df, ep_col):
    """Render budget breakdown pie and form timeline."""
    if current_squad_df.empty:
        return
    
    pos_spend = current_squad_df.groupby('position')['now_cost'].sum().to_dict()
    pos_order = ['GKP', 'DEF', 'MID', 'FWD']
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    bud1, bud2 = st.columns([1, 2])
    
    with bud1:
        st.markdown('<p class="section-title">Budget by Position</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(
            labels=[p for p in pos_order if p in pos_spend],
            values=[pos_spend.get(p, 0) for p in pos_order if p in pos_spend],
            marker_colors=[pos_colors[p] for p in pos_order if p in pos_spend],
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(color='#fff')
        )])
        fig_pie.update_layout(
            height=250,
            template='plotly_dark',
            paper_bgcolor='#0a0a0b',
            plot_bgcolor='#111113',
            font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True, key='opt_budget_pie')
    
    with bud2:
        st.markdown('<p class="section-title">Form Timeline</p>', unsafe_allow_html=True)
        form_data = current_squad_df.nlargest(8, 'form')[['web_name', 'form']].copy()
        
        fig_form = go.Figure()
        fig_form.add_trace(go.Bar(
            x=form_data['web_name'],
            y=form_data['form'],
            name='Form',
            marker_color='#ef4444',
            text=form_data['form'].round(1),
            textposition='outside'
        ))
        fig_form.update_layout(
            height=250,
            template='plotly_dark',
            paper_bgcolor='#0a0a0b',
            plot_bgcolor='#111113',
            font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
            margin=dict(l=40, r=20, t=20, b=60),
            xaxis_tickangle=-45,
            yaxis_title='Form'
        )
        st.plotly_chart(fig_form, use_container_width=True, key='opt_form_timeline')


def render_transfer_recommendations(current_squad_df, available_df, ep_col):
    """Render transfer IN/OUT recommendations."""
    if not current_squad_df.empty:
        st.markdown("**Recommended OUT** (lowest score in your squad)", unsafe_allow_html=True)
        out_candidates = current_squad_df.nsmallest(5, 'transfer_score')
        out_display = out_candidates[['web_name', 'team_name', 'position', 'now_cost', ep_col, 'form', 'transfer_score']].copy()
        out_display.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Form', 'Score']
        st.dataframe(style_df_with_injuries(out_display), hide_index=True, use_container_width=True)
        
        st.markdown("**Recommended IN** (best available)", unsafe_allow_html=True)


def render_ai_transfer_plan(current_squad_df, available_df, ep_col, free_transfers, bank):
    """Render AI-generated transfer plan."""
    st.markdown('<p class="section-title">AI Transfer Plan</p>', unsafe_allow_html=True)
    st.caption(f"Best {free_transfers} transfer(s) respecting FPL rules and budget")
    
    squad_value = current_squad_df['now_cost'].sum()
    total_budget = squad_value + bank
    
    out_suggestions = current_squad_df.nsmallest(free_transfers, 'transfer_score')
    
    transfer_plan = []
    remaining_budget = total_budget
    squad_teams = current_squad_df['team'].value_counts().to_dict()
    used_in_ids = set()
    
    for _, out_player in out_suggestions.iterrows():
        pos = out_player['position']
        out_cost = out_player['now_cost']
        
        pos_avail = available_df[
            (available_df['position'] == pos) &
            (~available_df['id'].isin(used_in_ids))
        ].copy()
        
        # Team limit check
        def team_ok(row):
            team = row['team']
            current_count = squad_teams.get(team, 0)
            if team == out_player['team']:
                return current_count - 1 < MAX_PLAYERS_PER_TEAM
            return current_count < MAX_PLAYERS_PER_TEAM
        
        pos_avail = pos_avail[pos_avail.apply(team_ok, axis=1)]
        
        max_spend = remaining_budget - (squad_value - out_cost)
        pos_avail = pos_avail[pos_avail['now_cost'] <= max_spend + 0.1]
        
        if not pos_avail.empty:
            best_in = pos_avail.nlargest(1, 'transfer_score').iloc[0]
            transfer_plan.append({
                'out_name': out_player['web_name'],
                'out_team': out_player.get('team_name', ''),
                'out_pos': pos,
                'out_price': out_cost,
                'out_ep': out_player[ep_col],
                'out_score': out_player['transfer_score'],
                'in_name': best_in['web_name'],
                'in_team': best_in.get('team_name', ''),
                'in_pos': pos,
                'in_price': best_in['now_cost'],
                'in_ep': best_in[ep_col],
                'in_score': best_in['transfer_score'],
            })
            used_in_ids.add(best_in['id'])
            squad_teams[out_player['team']] = squad_teams.get(out_player['team'], 1) - 1
            squad_teams[best_in['team']] = squad_teams.get(best_in['team'], 0) + 1
            remaining_budget = remaining_budget - best_in['now_cost'] + out_cost
    
    if transfer_plan:
        for t in transfer_plan:
            tc1, tc2, tc3 = st.columns([5, 1, 5])
            with tc1:
                st.markdown(f'''<div class="rule-card">
                    <div style="color:#ef4444;font-weight:600;">OUT: {t['out_name']}</div>
                    <div class="rule-label">{t['out_team']} | {t['out_pos']} | {t['out_price']:.1f}m | EP {t['out_ep']:.1f}</div>
                </div>''', unsafe_allow_html=True)
            with tc2:
                st.markdown('<div style="text-align:center;padding-top:1rem;color:#fff;font-size:1.5rem;">→</div>', unsafe_allow_html=True)
            with tc3:
                st.markdown(f'''<div class="rule-card">
                    <div style="color:#22c55e;font-weight:600;">IN: {t['in_name']}</div>
                    <div class="rule-label">{t['in_team']} | {t['in_pos']} | {t['in_price']:.1f}m | EP {t['in_ep']:.1f}</div>
                </div>''', unsafe_allow_html=True)
        
        # Summary
        total_out_cost = sum(t['out_price'] for t in transfer_plan)
        total_in_cost = sum(t['in_price'] for t in transfer_plan)
        net_spend = total_in_cost - total_out_cost
        ep_gain = sum(t['in_ep'] - t['out_ep'] for t in transfer_plan)
        
        st.markdown(f'''<div class="rule-card" style="margin-top:0.5rem;">
            <span style="color:#888;">Net Spend:</span> <span style="color:{'#ef4444' if net_spend > 0 else '#22c55e'};font-weight:600;">{net_spend:+.1f}m</span>
            &nbsp;|&nbsp;
            <span style="color:#888;">Bank After:</span> <span style="color:#fff;font-weight:600;">{bank - net_spend:.1f}m</span>
            &nbsp;|&nbsp;
            <span style="color:#888;">EP Change:</span> <span style="color:{'#22c55e' if ep_gain > 0 else '#ef4444'};font-weight:600;">{ep_gain:+.1f}</span>
        </div>''', unsafe_allow_html=True)
    else:
        st.info("Could not generate transfer plan. Budget may be too tight.")


def render_position_recommendations(available_df, ep_col):
    """Render recommendations by position."""
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_df = available_df[available_df['position'] == pos].nlargest(5, 'transfer_score')
        if not pos_df.empty:
            with st.expander(f"{pos} Recommendations", expanded=True):
                display_df = pos_df[['web_name', 'team_name', 'now_cost', ep_col, 'form', 'selected_by_percent', 'transfer_score']].copy()
                display_df.columns = ['Player', 'Team', 'Price', 'EP', 'Form', 'Owned%', 'Score']
                st.dataframe(style_df_with_injuries(display_df), hide_index=True, use_container_width=True)


def render_top_picks(available_df, ep_col):
    """Render top 10 overall picks."""
    st.markdown('<p class="section-title">Top 10 Overall Picks</p>', unsafe_allow_html=True)
    top_10 = available_df.nlargest(10, 'transfer_score')
    top_display = top_10[['web_name', 'team_name', 'position', 'now_cost', ep_col, 'form', 'selected_by_percent', 'transfer_score']].copy()
    top_display.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Form', 'Owned%', 'Score']
    st.dataframe(style_df_with_injuries(top_display), hide_index=True, use_container_width=True)


def render_points_projection(current_squad_df, available_df, ep_col):
    """Render points projection graph."""
    st.markdown('<p class="section-title">Points Projection (Next 5 GWs)</p>', unsafe_allow_html=True)
    
    current_ep_sum = current_squad_df[ep_col].sum()
    
    # Project with decay
    gws = list(range(1, 6))
    decay = [0.95 ** i for i in range(5)]
    current_proj = [current_ep_sum * d for d in decay]
    
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(
        x=[f'GW+{g}' for g in gws],
        y=current_proj,
        name='Current Squad',
        marker_color='#ef4444'
    ))
    
    fig_proj.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_title='Expected Points'
    )
    st.plotly_chart(fig_proj, use_container_width=True, key='opt_points_projection')
