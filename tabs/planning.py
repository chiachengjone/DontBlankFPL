"""Planning tab for FPL Strategy Engine - Chips and BGW/DGW Radar."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import safe_numeric


def render_planning_tab(processor, players_df: pd.DataFrame):
    """Planning tab - Chip strategy and Blank/Double GW planning."""
    
    try:
        current_gw = processor.fetcher.get_current_gameweek()
    except:
        current_gw = 1
    
    # Chip strategy
    render_chip_strategy(current_gw)
    
    # Blank/Double GW radar
    render_blank_double_gw_radar(processor, current_gw)


def render_chip_strategy(current_gw: int):
    """Render chip strategy planner."""
    st.markdown('<p class="section-title">Chip Strategy Planner</p>', unsafe_allow_html=True)
    st.caption("Optimal timing recommendations for your remaining chips")
    
    chip_cols = st.columns(4)
    
    with chip_cols[0]:
        st.markdown("**🃏 Wildcard**")
        st.markdown('''
        Best used when:
        - Squad value tanking
        - Major fixture swing
        - 5+ transfers needed
        ''')
        st.markdown(f"*Suggested: GW{min(current_gw + 3, 38)}*")
    
    with chip_cols[1]:
        st.markdown("**🆓 Free Hit**")
        st.markdown('''
        Best used when:
        - Blank gameweek
        - Double gameweek  
        - Fixture pileup
        ''')
        st.markdown("*Save for BGW/DGW*")
    
    with chip_cols[2]:
        st.markdown("**📈 Bench Boost**")
        st.markdown('''
        Best used when:
        - Strong bench
        - Double gameweek
        - All 15 play
        ''')
        st.markdown("*Combine with WC*")
    
    with chip_cols[3]:
        st.markdown("**👑 Triple Captain**")
        st.markdown('''
        Best used when:
        - DGW fixture
        - Easy home games
        - Premium in form
        ''')
        st.markdown("*Premium on DGW*")
    
    # Chip usage checklist
    st.markdown('<p class="section-title">Chip Status Tracker</p>', unsafe_allow_html=True)
    
    chk1, chk2, chk3, chk4 = st.columns(4)
    with chk1:
        wc1 = st.checkbox("Wildcard 1 Used", key="wc1_used")
        wc2 = st.checkbox("Wildcard 2 Used", key="wc2_used")
    with chk2:
        fh = st.checkbox("Free Hit Used", key="fh_used")
    with chk3:
        bb = st.checkbox("Bench Boost Used", key="bb_used")
    with chk4:
        tc = st.checkbox("Triple Captain Used", key="tc_used")
    
    chips_remaining = 5 - sum([wc1, wc2, fh, bb, tc])
    st.info(f"Chips remaining: {chips_remaining}/5")


def render_blank_double_gw_radar(processor, current_gw: int):
    """Render blank and double GW radar heatmap."""
    st.markdown('<p class="section-title">Blank & Double GW Radar</p>', unsafe_allow_html=True)
    
    try:
        fixtures = processor.fixtures_df
        
        if fixtures is None or fixtures.empty:
            st.info("Fixture data unavailable")
            return
        
        fixtures['event'] = safe_numeric(fixtures.get('event', pd.Series([0]*len(fixtures))))
        upcoming = fixtures[fixtures['event'] > current_gw].copy()
        
        teams_df = processor.fetcher.teams_df
        if teams_df is None:
            st.info("Team data unavailable")
            return
        
        # Count games per team per GW
        team_games = {}
        for gw in range(current_gw + 1, min(current_gw + 6, 39)):
            gw_fixtures = upcoming[upcoming['event'] == gw]
            for _, team in teams_df.iterrows():
                team_id = team['id']
                if team_id not in team_games:
                    team_games[team_id] = {}
                games = (gw_fixtures['team_h'] == team_id).sum() + (gw_fixtures['team_a'] == team_id).sum()
                team_games[team_id][gw] = games
        
        # Create heatmap data
        gw_range = list(range(current_gw + 1, min(current_gw + 6, 39)))
        team_names = [t['short_name'] for _, t in teams_df.iterrows()]
        team_ids = [t['id'] for _, t in teams_df.iterrows()]
        
        z_data = []
        for tid in team_ids:
            row = [team_games.get(tid, {}).get(gw, 1) for gw in gw_range]
            z_data.append(row)
        
        # Color scale: 0 = red (blank), 1 = grey, 2 = green (double)
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"GW{gw}" for gw in gw_range],
            y=team_names,
            colorscale=[[0, '#ef4444'], [0.5, '#4a4a4a'], [1, '#22c55e']],
            zmin=0,
            zmax=2,
            text=[[str(v) for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=12, color='white'),
            hovertemplate='%{y}: %{text} game(s) in %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            paper_bgcolor='#0a0a0b',
            plot_bgcolor='#111113',
            font=dict(family='Inter, sans-serif', color='#6b6b6b', size=10),
            margin=dict(l=60, r=20, t=20, b=40),
            xaxis=dict(side='top')
        )
        st.plotly_chart(fig, use_container_width=True, key='planning_bgw_dgw_heatmap')
        st.caption("🔴 0 = Blank GW  |  ⚫ 1 = Normal  |  🟢 2 = Double GW")
        
        # Summary stats
        blanks = []
        doubles = []
        for tid, games in team_games.items():
            team_name = teams_df[teams_df['id'] == tid]['short_name'].values
            team_name = team_name[0] if len(team_name) > 0 else 'UNK'
            for gw, count in games.items():
                if count == 0:
                    blanks.append({'Team': team_name, 'GW': gw})
                elif count >= 2:
                    doubles.append({'Team': team_name, 'GW': gw, 'Games': count})
        
        sum1, sum2 = st.columns(2)
        
        with sum1:
            st.markdown("**Upcoming Blanks**")
            if blanks:
                blank_df = pd.DataFrame(blanks)
                st.dataframe(blank_df, hide_index=True, use_container_width=True)
            else:
                st.success("No blanks in next 5 GWs")
        
        with sum2:
            st.markdown("**Upcoming Doubles**")
            if doubles:
                double_df = pd.DataFrame(doubles)
                st.dataframe(double_df, hide_index=True, use_container_width=True)
            else:
                st.info("No doubles in next 5 GWs")
        
    except Exception as e:
        st.info(f"Fixture data unavailable: {e}")
