"""Strategy tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import safe_numeric, get_injury_status, style_df_with_injuries, round_df
from components.charts import create_ep_ownership_scatter
from components.cards import render_player_detail_card
from fpl_api import CBIT_BONUS_THRESHOLD, CBIT_BONUS_POINTS, MAX_FREE_TRANSFERS, CAPTAIN_MULTIPLIER


def render_strategy_tab(processor, players_df: pd.DataFrame):
    """Strategy tab - EP vs Ownership visualization with filters."""
    
    # Season rules at top
    st.markdown('<p class="section-title">2025/26 Season Rules</p>', unsafe_allow_html=True)
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">{MAX_FREE_TRANSFERS}</div>
            <div class="rule-label">Max Free Transfers</div>
        </div>
        ''', unsafe_allow_html=True)
    with r2:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">{CAPTAIN_MULTIPLIER}x</div>
            <div class="rule-label">Captain Multiplier</div>
        </div>
        ''', unsafe_allow_html=True)
    with r3:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">+{CBIT_BONUS_POINTS}</div>
            <div class="rule-label">CBIT Bonus ({CBIT_BONUS_THRESHOLD} actions)</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Player Landscape</p>', unsafe_allow_html=True)
    
    # Filters
    f1, f2, f3, f4 = st.columns([1, 1, 1.5, 1.5])
    
    with f1:
        pos_filter = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="strat_pos")
    with f2:
        all_teams = sorted(players_df['team_name'].dropna().unique().tolist()) if 'team_name' in players_df.columns else []
        team_filter = st.selectbox("Team", ['All'] + all_teams, key="strat_team")
    with f3:
        max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="strat_price")
    with f4:
        search_player = st.text_input("Search player", placeholder="Type name to highlight...", key="strat_search")
    
    # Use base player data for fast graph rendering
    df = players_df.copy()
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df = df[df['now_cost'] <= max_price]
    
    if pos_filter != 'All':
        df = df[df['position'] == pos_filter]
    
    if team_filter != 'All' and 'team_name' in df.columns:
        df = df[df['team_name'] == team_filter]
    
    # Player detail view when searching
    if search_player and search_player.strip():
        search_lower = search_player.lower().strip()
        matched = players_df[players_df['web_name'].str.lower().str.contains(search_lower, na=False)]
        
        if not matched.empty:
            exact = matched[matched['web_name'].str.lower() == search_lower]
            selected = exact.iloc[0] if not exact.empty else matched.iloc[0]
            
            st.markdown('<p class="section-title">Player Details</p>', unsafe_allow_html=True)
            render_player_detail_card(selected, processor, players_df)
            
            if len(matched) > 1:
                st.caption(f"Other matches: {', '.join(matched['web_name'].tolist()[:10])}")
    
    # Scatter plot
    if 'expected_points' in df.columns or 'ep_next' in df.columns:
        fig = create_ep_ownership_scatter(df, pos_filter, search_player=search_player)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key='strategy_ep_ownership_scatter')
    else:
        st.info("Data loading...")
    
    # Quick stats
    st.markdown('<p class="section-title">Quick Stats</p>', unsafe_allow_html=True)
    render_quick_stats(df)
    
    # Form vs EP bubble chart
    render_form_vs_ep_chart(df)
    
    # Captain planning
    render_captain_planning(players_df, processor)
    
    # Fixture difficulty
    render_fixture_difficulty(processor)
    
    # Price watch
    render_price_watch(players_df)
    
    # Injury alerts
    render_injury_alerts(players_df)
    
    # Ownership trends
    render_ownership_trends(players_df)


def render_quick_stats(df: pd.DataFrame):
    """Render quick stats row."""
    stat1, stat2, stat3, stat4 = st.columns(4)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    with stat1:
        top_owned = df.nlargest(1, 'selected_by_percent')
        if not top_owned.empty:
            st.metric("Most Owned", top_owned.iloc[0]['web_name'], f"{top_owned.iloc[0]['selected_by_percent']:.1f}%")
    
    with stat2:
        if 'ep_next' in df.columns:
            df['ep_next'] = safe_numeric(df['ep_next'])
            top_ep = df.nlargest(1, 'ep_next')
            if not top_ep.empty:
                st.metric("Top EP", top_ep.iloc[0]['web_name'], f"{top_ep.iloc[0]['ep_next']:.1f}")
    
    with stat3:
        st.metric("Players Shown", len(df))
    
    with stat4:
        avg_price = df['now_cost'].mean()
        st.metric("Avg Price", f"{avg_price:.1f}m")


def render_form_vs_ep_chart(df: pd.DataFrame):
    """Render Form vs EP bubble chart — size = price, color = position."""
    st.markdown('<p class="section-title">Form vs Expected Points</p>', unsafe_allow_html=True)
    st.caption("Bubble size = price. Top-right = hot and high-ceiling players.")
    
    chart_df = df.copy()
    chart_df['form'] = safe_numeric(chart_df.get('form', pd.Series([0]*len(chart_df))))
    chart_df['ep'] = safe_numeric(chart_df.get('ep_next', chart_df.get('expected_points', pd.Series([2]*len(chart_df)))))
    chart_df['minutes'] = safe_numeric(chart_df.get('minutes', pd.Series([0]*len(chart_df))))
    chart_df = chart_df[chart_df['minutes'] > 200]
    
    if chart_df.empty or len(chart_df) < 5:
        return
    
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    fig = go.Figure()
    for pos, color in pos_colors.items():
        pos_df = chart_df[chart_df['position'] == pos]
        if pos_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pos_df['form'],
            y=pos_df['ep'],
            mode='markers',
            name=pos,
            marker=dict(
                size=pos_df['now_cost'].clip(lower=4, upper=14) * 2.5,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.15)')
            ),
            text=pos_df['web_name'],
            hovertemplate='<b>%{text}</b><br>Form: %{x:.1f}<br>EP: %{y:.1f}<br><extra></extra>'
        ))
    
    # Add quadrant lines at medians
    med_form = chart_df['form'].median()
    med_ep = chart_df['ep'].median()
    
    fig.add_hline(y=med_ep, line_dash="dot", line_color="rgba(255,255,255,0.08)")
    fig.add_vline(x=med_form, line_dash="dot", line_color="rgba(255,255,255,0.08)")
    
    fig.update_layout(
        height=420,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        xaxis=dict(title='Form', gridcolor='#1e1e21', zerolinecolor='#1e1e21'),
        yaxis=dict(title='Expected Points', gridcolor='#1e1e21', zerolinecolor='#1e1e21'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=30, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='strategy_form_vs_ep')


def render_captain_planning(players_df: pd.DataFrame, processor):
    """Render captain planning section."""
    st.markdown('<p class="section-title">Captain Picks</p>', unsafe_allow_html=True)
    st.caption("Top captain options based on EP, form, and fixture difficulty")
    
    cap_df = players_df.copy()
    cap_df['ep_next'] = safe_numeric(cap_df.get('ep_next', pd.Series([0]*len(cap_df))))
    cap_df['form'] = safe_numeric(cap_df.get('form', pd.Series([0]*len(cap_df))))
    cap_df['selected_by_percent'] = safe_numeric(cap_df['selected_by_percent'])
    cap_df['minutes'] = safe_numeric(cap_df.get('minutes', pd.Series([0]*len(cap_df))))
    cap_df = cap_df[cap_df['minutes'] > 500]
    
    cap_df['captain_score'] = cap_df['ep_next'] * CAPTAIN_MULTIPLIER + cap_df['form'] * 0.3
    cap_df['captain_ev'] = cap_df['ep_next'] * CAPTAIN_MULTIPLIER
    
    cap_cols = st.columns(5)
    top_caps = cap_df.nlargest(5, 'captain_score')
    
    for i, (_, cap) in enumerate(top_caps.iterrows()):
        with cap_cols[i]:
            injury = get_injury_status(cap)
            injury_badge = f'<span style="color:{injury["color"]}"> [{injury["icon"]}]</span>' if injury['icon'] else ''
            st.markdown(f'''
            <div class="rule-card">
                <div style="font-size:0.75rem;color:#888;">#{i+1}</div>
                <div style="font-size:1.1rem;font-weight:600;color:#fff;">{cap["web_name"]}{injury_badge}</div>
                <div style="color:#888;font-size:0.85rem;">{cap.get("team_name", "")} | {cap["now_cost"]:.1f}m</div>
                <div style="color:#ef4444;font-weight:600;margin-top:0.5rem;">{cap["captain_ev"]:.1f} EV</div>
                <div style="color:#888;font-size:0.75rem;">Form: {cap["form"]:.1f} | Own: {cap["selected_by_percent"]:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)


def render_fixture_difficulty(processor):
    """Render team fixture difficulty heatmap."""
    st.markdown('<p class="section-title">Team Fixture Difficulty</p>', unsafe_allow_html=True)
    
    try:
        current_gw = processor.fetcher.get_current_gameweek()
        fixtures_df = processor.fixtures_df
        teams_df = processor.teams_df
        
        team_fixtures = []
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_fix = fixtures_df[
                (fixtures_df['event'] >= current_gw) &
                (fixtures_df['event'] < current_gw + 6) &
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
            ].sort_values('event')
            
            fdrs = []
            for _, fix in team_fix.iterrows():
                is_home = fix['team_h'] == team_id
                fdr = fix.get('team_h_difficulty', 3) if is_home else fix.get('team_a_difficulty', 3)
                fdrs.append(int(fdr))
            
            while len(fdrs) < 6:
                fdrs.append(3)
            
            team_fixtures.append({
                'Team': team.get('short_name', team['name']),
                'Avg FDR': round(np.mean(fdrs[:6]), 2),
                **{f'GW{current_gw + i}': fdrs[i] for i in range(6)}
            })
        
        team_fix_df = pd.DataFrame(team_fixtures).sort_values('Avg FDR')
        fdr_cols = [f'GW{current_gw + i}' for i in range(6)]
        
        fig = go.Figure(data=go.Heatmap(
            z=team_fix_df[fdr_cols].values,
            x=fdr_cols,
            y=team_fix_df['Team'],
            colorscale=[[0, '#22c55e'], [0.25, '#77c45e'], [0.5, '#f59e0b'], [0.75, '#ef6b4e'], [1, '#ef4444']],
            showscale=True,
            colorbar=dict(title='FDR', tickvals=[1, 2, 3, 4, 5]),
            zmin=1, zmax=5
        ))
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            paper_bgcolor='#0a0a0b',
            plot_bgcolor='#111113',
            font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
            margin=dict(l=80, r=40, t=20, b=40),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig, use_container_width=True, key='strategy_fixture_heatmap')
        
    except Exception as e:
        st.info(f"Fixture data unavailable: {e}")


def render_price_watch(players_df: pd.DataFrame):
    """Render price watch section."""
    st.markdown('<p class="section-title">Price Watch</p>', unsafe_allow_html=True)
    
    price_df = players_df.copy()
    price_df['transfers_in_event'] = safe_numeric(price_df.get('transfers_in_event', pd.Series([0]*len(price_df))))
    price_df['transfers_out_event'] = safe_numeric(price_df.get('transfers_out_event', pd.Series([0]*len(price_df))))
    price_df['net_transfers'] = price_df['transfers_in_event'] - price_df['transfers_out_event']
    price_df['minutes'] = safe_numeric(price_df.get('minutes', pd.Series([0]*len(price_df))))
    price_df = price_df[price_df['minutes'] > 0]
    
    pr1, pr2 = st.columns(2)
    
    with pr1:
        st.markdown("**Likely to Rise**")
        risers = price_df.nlargest(8, 'net_transfers')[['web_name', 'team_name', 'now_cost', 'net_transfers']]
        risers.columns = ['Player', 'Team', 'Price', 'Net Transfers']
        risers['Net Transfers'] = risers['Net Transfers'].apply(lambda x: f"+{int(x):,}")
        st.dataframe(style_df_with_injuries(risers), hide_index=True, use_container_width=True)
    
    with pr2:
        st.markdown("**Likely to Fall**")
        fallers = price_df.nsmallest(8, 'net_transfers')[['web_name', 'team_name', 'now_cost', 'net_transfers']]
        fallers.columns = ['Player', 'Team', 'Price', 'Net Transfers']
        fallers['Net Transfers'] = fallers['Net Transfers'].apply(lambda x: f"{int(x):,}")
        st.dataframe(style_df_with_injuries(fallers), hide_index=True, use_container_width=True)


def render_injury_alerts(players_df: pd.DataFrame):
    """Render injury alerts section."""
    st.markdown('<p class="section-title">Injury & Doubt Alerts</p>', unsafe_allow_html=True)
    
    df = players_df.copy()
    df['chance_of_playing'] = safe_numeric(df.get('chance_of_playing_next_round', pd.Series([100]*len(df))), 100)
    df['status'] = df.get('status', 'a')
    df['news'] = df.get('news', '').fillna('')
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    flagged = df[
        ((df['status'].isin(['i', 'd', 's', 'u'])) | (df['chance_of_playing'] < 100)) &
        (df['selected_by_percent'] > 1)
    ].sort_values('selected_by_percent', ascending=False)
    
    if not flagged.empty:
        display = flagged.head(15)[['web_name', 'team_name', 'position', 'selected_by_percent', 'chance_of_playing', 'news']].copy()
        display = display.reset_index(drop=True)
        display.columns = ['Player', 'Team', 'Pos', 'Own%', 'Chance', 'News']
        display['Own%'] = display['Own%'].round(1)
        # Store numeric chance for styling before converting to string
        chance_values = display['Chance'].astype(int).tolist()
        display['Chance'] = display['Chance'].astype(int).astype(str) + '%'
        
        # Style rows based on injury severity
        def style_injury_rows(df):
            styles = []
            for i in range(len(df)):
                chance = chance_values[i] if i < len(chance_values) else 100
                if chance == 0:
                    bg = 'background-color: rgba(239, 68, 68, 0.4)'  # Red - out
                elif chance <= 25:
                    bg = 'background-color: rgba(239, 68, 68, 0.3)'  # Dark red
                elif chance <= 50:
                    bg = 'background-color: rgba(249, 115, 22, 0.3)'  # Orange
                elif chance <= 75:
                    bg = 'background-color: rgba(245, 158, 11, 0.25)'  # Amber
                else:
                    bg = 'background-color: rgba(250, 204, 21, 0.15)'  # Light yellow
                styles.append([bg] * len(df.columns))
            return pd.DataFrame(styles, index=df.index, columns=df.columns)
        
        styled = display.style.apply(style_injury_rows, axis=None)
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.success("No major injury concerns in popular players")


def render_ownership_trends(players_df: pd.DataFrame):
    """Render ownership trends chart."""
    st.markdown('<p class="section-title">Ownership Trends</p>', unsafe_allow_html=True)
    
    df = players_df.copy()
    df['transfers_in_event'] = safe_numeric(df.get('transfers_in_event', pd.Series([0]*len(df))))
    df['transfers_out_event'] = safe_numeric(df.get('transfers_out_event', pd.Series([0]*len(df))))
    df['total_transfers'] = df['transfers_in_event'] + df['transfers_out_event']
    
    top_movers = df.nlargest(20, 'total_transfers').copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=top_movers['transfers_in_event'],
        name='In',
        marker_color='#22c55e'
    ))
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=-top_movers['transfers_out_event'],
        name='Out',
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        barmode='relative',
        height=350,
        template='plotly_dark',
        paper_bgcolor='#0a0a0b',
        plot_bgcolor='#111113',
        font=dict(family='Inter, sans-serif', color='#6b6b6b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=80),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True, key='strategy_ownership_trends')
