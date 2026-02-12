"""Rival Scout tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import safe_numeric, style_df_with_injuries
from fpl_api import RivalScout


def render_rival_tab(processor, players_df: pd.DataFrame):
    """Rival Scout tab content."""
    
    st.markdown('<p class="section-title">Rival Comparison</p>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        your_id = st.number_input("Your Team ID", min_value=1, max_value=10000000, value=1, key="your_id")
    with c2:
        rival_id = st.number_input("Rival Team ID", min_value=1, max_value=10000000, value=2, key="rival_id")
    
    if st.button("Compare Teams", type="primary", use_container_width=True):
        if your_id == rival_id:
            st.warning("Enter different Team IDs")
            return
        
        with st.spinner("Analyzing..."):
            try:
                scout = RivalScout(processor.fetcher, processor)
                result = scout.compare_teams(your_id, rival_id)
                
                if result['valid']:
                    render_comparison_metrics(result, players_df)
                    render_unique_players_chart(result, players_df)
                    render_unique_players_tables(result, players_df)
                else:
                    st.error(result.get('error', 'Comparison failed'))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter Team IDs and click Compare to analyze rival squads")


def render_comparison_metrics(result: dict, players_df: pd.DataFrame):
    """Render comparison metrics."""
    # Calculate total team EP
    your_ep = calculate_team_ep(result.get('your_squad', []), players_df)
    rival_ep = calculate_team_ep(result.get('rival_squad', []), players_df)
    
    # Win probability based on EP difference
    ep_diff = your_ep - rival_ep
    # Simple sigmoid-based probability
    import math
    win_prob = 1 / (1 + math.exp(-ep_diff / 10)) * 100
    
    st.markdown('<p class="section-title">Expected GW Score</p>', unsafe_allow_html=True)
    
    # Team EP comparison
    score1, score2, score3 = st.columns(3)
    
    with score1:
        st.markdown(f'''
        <div class="rule-card">
            <div style="color:#22c55e;font-size:2rem;font-weight:700;">{your_ep:.1f}</div>
            <div class="rule-label">Your Expected Points</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with score2:
        color = '#22c55e' if ep_diff > 0 else '#ef4444' if ep_diff < 0 else '#888'
        win_color = '#22c55e' if win_prob > 55 else '#ef4444' if win_prob < 45 else '#f59e0b'
        st.markdown(f'''
        <div class="rule-card">
            <div style="color:{win_color};font-size:2rem;font-weight:700;">{win_prob:.0f}%</div>
            <div class="rule-label">Your Win Probability</div>
            <div style="color:{color};font-size:0.9rem;margin-top:0.5rem;">EP Diff: {ep_diff:+.1f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with score3:
        st.markdown(f'''
        <div class="rule-card">
            <div style="color:#ef4444;font-size:2rem;font-weight:700;">{rival_ep:.1f}</div>
            <div class="rule-label">Rival Expected Points</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Additional metrics row
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Jaccard Similarity", f"{result['jaccard_similarity']*100:.1f}%")
    with m2:
        st.metric("Common Players", result['overlap_count'])
    with m3:
        biggest_threat = result.get('biggest_threat')
        if biggest_threat:
            st.metric("Biggest Threat", biggest_threat['name'], f"EP {biggest_threat['expected_points']:.1f}")
        else:
            st.metric("Biggest Threat", "None")


def calculate_team_ep(squad_ids: list, players_df: pd.DataFrame) -> float:
    """Calculate total expected points for a team."""
    total_ep = 0.0
    for pid in squad_ids[:11]:  # Starting 11
        p = players_df[players_df['id'] == pid]
        if not p.empty:
            ep = safe_numeric(pd.Series([p.iloc[0].get('ep_next', p.iloc[0].get('expected_points', 2))])).iloc[0]
            total_ep += ep
    return total_ep


def get_player_details(pid_list: list, label: str, players_df: pd.DataFrame) -> list:
    """Get detailed player info for a list of IDs."""
    rows = []
    for pid in pid_list:
        p = players_df[players_df['id'] == pid]
        if not p.empty:
            p = p.iloc[0]
            ep = safe_numeric(pd.Series([p.get('ep_next', p.get('expected_points', 0))])).iloc[0]
            form = safe_numeric(pd.Series([p.get('form', 0)])).iloc[0]
            pts = safe_numeric(pd.Series([p.get('total_points', 0)])).iloc[0]
            cost = safe_numeric(pd.Series([p.get('now_cost', 5)])).iloc[0]
            own = safe_numeric(pd.Series([p.get('selected_by_percent', 0)])).iloc[0]
            threat = round(ep * 0.5 + form * 0.3 + (pts / max(1, 38)) * 0.2, 2)
            rows.append({
                'Player': p.get('web_name', f'ID {pid}'),
                'Pos': p.get('position', '?'),
                'Team': p.get('team_name', ''),
                'Price': round(cost, 1),
                'EP': round(ep, 2),
                'Form': round(form, 1),
                'Pts': int(pts),
                'Own%': round(own, 1),
                'Threat': threat,
                'Side': label
            })
    return rows


def render_unique_players_chart(result: dict, players_df: pd.DataFrame):
    """Render unique players comparison chart."""
    your_details = get_player_details(result['unique_to_you'], 'You', players_df)
    rival_details = get_player_details(result['unique_to_rival'], 'Rival', players_df)
    all_unique = your_details + rival_details
    
    if not all_unique:
        return
    
    unique_df = pd.DataFrame(all_unique)
    
    st.markdown('<p class="section-title">Unique Players Comparison</p>', unsafe_allow_html=True)
    st.caption("Threat Score = EP x 0.5 + Form x 0.3 + Season Pts x 0.2 (normalized). Higher = more dangerous.")
    
    color_map = {'You': '#22c55e', 'Rival': '#ef4444'}
    fig = go.Figure()
    
    for side in ['You', 'Rival']:
        side_df = unique_df[unique_df['Side'] == side]
        if side_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=side_df['Player'],
            y=side_df['Threat'],
            name=f"{side} Unique",
            marker_color=color_map[side],
            text=side_df['Pos'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Threat: %{y:.2f}<br>Pos: %{text}<extra></extra>'
        ))
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ccc'),
        yaxis_title='Threat Score',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_unique_players_tables(result: dict, players_df: pd.DataFrame):
    """Render unique players tables."""
    your_details = get_player_details(result['unique_to_you'], 'You', players_df)
    rival_details = get_player_details(result['unique_to_rival'], 'Rival', players_df)
    
    tc1, tc2 = st.columns(2)
    
    with tc1:
        st.markdown("**Your Unique Players**")
        if your_details:
            your_df = pd.DataFrame(your_details)[['Player', 'Pos', 'Team', 'Price', 'EP', 'Form', 'Pts', 'Own%', 'Threat']]
            st.dataframe(style_df_with_injuries(your_df.sort_values('Threat', ascending=False)), hide_index=True, use_container_width=True)
        else:
            st.info("No unique players")
    
    with tc2:
        st.markdown("**Rival's Unique Players**")
        if rival_details:
            rival_df = pd.DataFrame(rival_details)[['Player', 'Pos', 'Team', 'Price', 'EP', 'Form', 'Pts', 'Own%', 'Threat']]
            st.dataframe(style_df_with_injuries(rival_df.sort_values('Threat', ascending=False)), hide_index=True, use_container_width=True)
        else:
            st.info("No unique players")
