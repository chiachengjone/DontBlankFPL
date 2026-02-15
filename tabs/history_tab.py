"""Historical Performance Tab - Track actual weekly scores and decisions."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric
from components.styles import render_section_title


def render_history_tab(processor, players_df: pd.DataFrame, fetcher):
    """Historical Performance tab - track scores and validate ML predictions."""
    
    st.markdown('<p class="section-title">Historical Performance</p>', unsafe_allow_html=True)
    st.caption("Track your GW scores, captain choices, and ML prediction accuracy")
    
    # Get team ID
    team_id = st.session_state.get('fpl_team_id', 0)
    
    if team_id == 0:
        st.warning("Enter your FPL Team ID in the header to view historical performance")
        
        # Show general performance analysis instead
        render_general_analysis(processor, players_df, fetcher)
        return
    
    # Try to fetch historical data
    try:
        with st.spinner("Loading your history..."):
            history = fetcher.get_team_history(team_id)
            current_gw = fetcher.get_current_gameweek()
    except Exception as e:
        st.error(f"Could not load history: {e}")
        render_general_analysis(processor, players_df, fetcher)
        return
    
    if not history or 'current' not in history:
        st.warning("No historical data found for this Team ID")
        render_general_analysis(processor, players_df, fetcher)
        return
    
    gw_history = history.get('current', [])
    
    if not gw_history:
        st.info("No gameweek history available yet")
        return
    
    # ── Season Summary ──
    st.markdown("### Season Summary")
    
    total_points = sum(gw.get('points', 0) for gw in gw_history)
    total_transfers = sum(gw.get('event_transfers', 0) for gw in gw_history)
    total_hits = sum(gw.get('event_transfers_cost', 0) for gw in gw_history)
    avg_gw_points = total_points / len(gw_history) if gw_history else 0
    current_rank = gw_history[-1].get('overall_rank', 0) if gw_history else 0
    best_gw = max(gw_history, key=lambda x: x.get('points', 0)) if gw_history else {}
    worst_gw = min(gw_history, key=lambda x: x.get('points', 0)) if gw_history else {}
    
    sum_cols = st.columns(4)
    
    with sum_cols[0]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Points</div>
            <div style="color:#3b82f6;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{total_points}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[1]:
        rank_color = '#22c55e' if current_rank < 100000 else '#f59e0b' if current_rank < 500000 else '#6b6b6b'
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Overall Rank</div>
            <div style="color:{rank_color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{current_rank:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[2]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg GW Points</div>
            <div style="color:#22c55e;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_gw_points:.1f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[3]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Transfer Hits</div>
            <div style="color:#ef4444;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">-{total_hits}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── GW-by-GW Performance Chart ──
    st.markdown("### Gameweek Performance")
    
    gws = [gw.get('event', i+1) for i, gw in enumerate(gw_history)]
    points = [gw.get('points', 0) for gw in gw_history]
    ranks = [gw.get('overall_rank', 0) for gw in gw_history]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Points bars
    fig.add_trace(
        go.Bar(x=gws, y=points, name='Points', marker_color='#3b82f6', opacity=0.8),
        secondary_y=False
    )
    
    # Average line
    avg_line = [avg_gw_points] * len(gws)
    fig.add_trace(
        go.Scatter(x=gws, y=avg_line, name='Average', line=dict(color='#22c55e', dash='dash'), mode='lines'),
        secondary_y=False
    )
    
    # Rank line
    fig.add_trace(
        go.Scatter(x=gws, y=ranks, name='Rank', line=dict(color='#f59e0b'), mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=50, t=40, b=50)
    )
    
    fig.update_xaxes(title_text='Gameweek', gridcolor='#e5e5ea')
    fig.update_yaxes(title_text='Points', gridcolor='#e5e5ea', secondary_y=False)
    fig.update_yaxes(title_text='Rank', gridcolor='#e5e5ea', secondary_y=True, autorange='reversed')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ── Best/Worst GWs ──
    st.markdown("### Highlights")
    
    hl_cols = st.columns(2)
    
    with hl_cols[0]:
        st.markdown("**Best Gameweek**")
        if best_gw:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #22c55e;border-radius:10px;padding:1rem;">
                <div style="color:#6ee7b7;font-size:0.72rem;font-weight:500;text-transform:uppercase;">GW{best_gw.get('event', '?')}</div>
                <div style="color:#22c55e;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{best_gw.get('points', 0)} pts</div>
                <div style="color:#86868b;font-size:0.8rem;">Rank: {best_gw.get('rank', 'N/A'):,}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with hl_cols[1]:
        st.markdown("**Worst Gameweek**")
        if worst_gw:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #ef4444;border-radius:10px;padding:1rem;">
                <div style="color:#fca5a5;font-size:0.72rem;font-weight:500;text-transform:uppercase;">GW{worst_gw.get('event', '?')}</div>
                <div style="color:#ef4444;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{worst_gw.get('points', 0)} pts</div>
                <div style="color:#86868b;font-size:0.8rem;">Rank: {worst_gw.get('rank', 'N/A'):,}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Transfer Analysis ──
    st.markdown("### Transfer Analysis")
    
    transfers_per_gw = [gw.get('event_transfers', 0) for gw in gw_history]
    hits_per_gw = [gw.get('event_transfers_cost', 0) for gw in gw_history]
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=gws, y=transfers_per_gw, name='Transfers',
        marker_color='#3b82f6'
    ))
    
    fig2.add_trace(go.Bar(
        x=gws, y=[-h/4 for h in hits_per_gw], name='Hits (cost/4)',
        marker_color='#ef4444'
    ))
    
    fig2.update_layout(
        height=250,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        barmode='relative',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(title='Gameweek', gridcolor='#e5e5ea'),
        yaxis=dict(title='Transfers', gridcolor='#e5e5ea'),
        margin=dict(l=50, r=30, t=40, b=50)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Transfer ROI analysis
    st.markdown("**Transfer Hit ROI**")
    
    gws_with_hits = [(gw, gw.get('event_transfers_cost', 0), gw.get('points', 0)) 
                     for gw in gw_history if gw.get('event_transfers_cost', 0) > 0]
    
    if gws_with_hits:
        hit_df = pd.DataFrame([{
            'GW': gw.get('event', '?'),
            'Hit Cost': cost,
            'Points Scored': pts,
            'Net Points': pts - cost,
            'Worth It?': 'Yes' if pts > avg_gw_points + cost else 'No'
        } for gw, cost, pts in gws_with_hits])
        
        st.dataframe(hit_df, hide_index=True, use_container_width=True)
        
        hits_worth_it = sum(1 for gw, cost, pts in gws_with_hits if pts > avg_gw_points + cost)
        st.caption(f"Hits worth it: {hits_worth_it}/{len(gws_with_hits)} ({hits_worth_it/len(gws_with_hits)*100:.0f}%)")
    else:
        st.success("No transfer hits taken this season!")
    
    st.markdown("---")
    
    # ── Rank Movement ──
    st.markdown("### Rank Movement")
    
    rank_changes = []
    for i in range(1, len(gw_history)):
        prev_rank = gw_history[i-1].get('overall_rank', 0)
        curr_rank = gw_history[i].get('overall_rank', 0)
        if prev_rank > 0 and curr_rank > 0:
            change = prev_rank - curr_rank  # Positive = improvement
            rank_changes.append({
                'gw': gw_history[i].get('event', i+1),
                'change': change,
                'pts': gw_history[i].get('points', 0)
            })
    
    if rank_changes:
        green_arrows = sum(1 for rc in rank_changes if rc['change'] > 0)
        red_arrows = sum(1 for rc in rank_changes if rc['change'] < 0)
        
        arrow_cols = st.columns(2)
        with arrow_cols[0]:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #22c55e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{green_arrows}</div>
                <div style="color:#86868b;font-size:0.72rem;text-transform:uppercase;">Green Arrows</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with arrow_cols[1]:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #ef4444;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#ef4444;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{red_arrows}</div>
                <div style="color:#86868b;font-size:0.72rem;text-transform:uppercase;">Red Arrows</div>
            </div>
            ''', unsafe_allow_html=True)


def render_general_analysis(processor, players_df: pd.DataFrame, fetcher):
    """Render general analysis when no team ID is provided."""
    
    st.markdown("### General Performance Analysis")
    st.caption("Enter your Team ID to see personalized statistics")
    
    df = players_df.copy()
    df['total_points'] = safe_numeric(df.get('total_points', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    # Top scorers this season
    st.markdown("#### Top Scorers This Season")
    
    top_scorers = df.nlargest(10, 'total_points')[
        ['web_name', 'team_name', 'position', 'now_cost', 'total_points', 'selected_by_percent']
    ].copy()
    top_scorers.columns = ['Player', 'Team', 'Pos', 'Price', 'Total Points', 'EO%']
    
    st.dataframe(
        top_scorers.style.format({
            'Price': '£{:.1f}m',
            'Total Points': '{:.0f}',
            'EO%': '{:.1f}%'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    # Points per million leaders
    st.markdown("#### Best Value (Points Per Million)")
    
    df['ppm'] = df['total_points'] / df['now_cost'].clip(lower=4)
    ppm_leaders = df[df['total_points'] > 30].nlargest(10, 'ppm')[
        ['web_name', 'team_name', 'position', 'now_cost', 'total_points', 'ppm']
    ].copy()
    ppm_leaders.columns = ['Player', 'Team', 'Pos', 'Price', 'Total Pts', 'Pts/£m']
    
    st.dataframe(
        ppm_leaders.style.format({
            'Price': '£{:.1f}m',
            'Total Pts': '{:.0f}',
            'Pts/£m': '{:.2f}'
        }),
        hide_index=True,
        use_container_width=True
    )
