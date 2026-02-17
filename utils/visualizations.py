"""
Visualization utilities for FPL Strategy Engine.
"""

import streamlit as st
import pandas as pd

def render_pitch_view(squad_df: pd.DataFrame, title: str = "Optimized Squad"):
    """
    Render a football pitch visualization for the squad.
    
    Args:
        squad_df: DataFrame containing player data. 
                  Must have columns: 'web_name', 'position', 'now_cost', 'expected_points', 'team_short_name', 'Status'
    """
    st.markdown(f"### {title}")
    
    # Separate starters and bench
    starters = squad_df[squad_df['Status'].isin(['XI', '(C)', '(V)'])]
    bench = squad_df[squad_df['Status'] == 'Bench']
    
    # Sort starters by position standard order
    pos_order = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    starters['pos_order'] = starters['position'].map(pos_order)
    starters = starters.sort_values(['pos_order', 'expected_points'], ascending=[True, False])
    
    # Group by position
    gkp = starters[starters['position'] == 'GKP']
    defs = starters[starters['position'] == 'DEF']
    mids = starters[starters['position'] == 'MID']
    fwds = starters[starters['position'] == 'FWD']
    
    # CSS for pitch
    st.markdown("""
    <style>
    .pitch-container {
        background-color: #1a4f1a; /* Dark Green */
        background-image: 
            linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent),
            linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, .05) 25%, rgba(255, 255, 255, .05) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .05) 75%, rgba(255, 255, 255, .05) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        position: relative;
        border: 2px solid #fff;
    }
    .pitch-row {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 25px;
        gap: 20px;
    }
    .player-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 6px;
        padding: 6px;
        width: 90px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .player-card:hover {
        transform: translateY(-2px);
        background-color: #fff;
        z-index: 10;
    }
    .player-name {
        font-weight: 700;
        font-size: 0.85rem;
        color: #1e293b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }
    .player-team {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
    }
    .player-meta {
        display: flex;
        justify-content: center;
        gap: 6px;
        font-size: 0.75rem;
        margin-top: 2px;
        border-top: 1px solid #e2e8f0;
        padding-top: 2px;
    }
    .player-xp {
        color: #22c55e;
        font-weight: 600;
    }
    .player-cost {
        color: #64748b;
    }
    .captain-badge {
        background-color: #ef4444;
        color: white;
        font-size: 0.6rem;
        padding: 1px 4px;
        border-radius: 4px;
        margin-left: 4px;
        vertical-align: middle;
    }
    .bench-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        padding: 15px;
        background-color: #f1f5f9;
        border-radius: 8px;
        border: 1px dashed #cbd5e1;
    }
    .bench-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-align: center;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    def _render_player(row, is_captain=False, is_vice=False):
        badge = ""
        if is_captain:
            badge = '<span class="captain-badge">C</span>'
        elif is_vice:
            badge = '<span class="captain-badge" style="background-color:#3b82f6;">V</span>'
            
        xp = row.get('expected_points', 0)
        # Handle if xp is string or float
        try:
            xp_str = f"{float(xp):.1f}"
        except:
            xp_str = str(xp)
            
        return f"""
        <div class="player-card">
            <div class="player-name">{row['web_name']}{badge}</div>
            <div class="player-team">{row.get('team_short_name', 'UNK')} • {row['position']}</div>
            <div class="player-meta">
                <span class="player-xp">{xp_str} xP</span>
                <span class="player-cost">£{row['now_cost']}</span>
            </div>
        </div>
        """

    # Pitch Rows
    html = '<div class="pitch-container">'
    
    # GKP
    html += '<div class="pitch-row">'
    for _, p in gkp.iterrows():
        html += _render_player(p, p['Status']=='(C)', p['Status']=='(V)')
    html += '</div>'
    
    # DEF
    html += '<div class="pitch-row">'
    for _, p in defs.iterrows():
        html += _render_player(p, p['Status']=='(C)', p['Status']=='(V)')
    html += '</div>'
    
    # MID
    html += '<div class="pitch-row">'
    for _, p in mids.iterrows():
        html += _render_player(p, p['Status']=='(C)', p['Status']=='(V)')
    html += '</div>'
    
    # FWD
    html += '<div class="pitch-row">'
    for _, p in fwds.iterrows():
        html += _render_player(p, p['Status']=='(C)', p['Status']=='(V)')
    html += '</div>'
    
    html += '</div>' # End pitch-container
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Bench
    st.markdown('<div class="bench-label">BENCH</div>', unsafe_allow_html=True)
    bench_html = '<div class="bench-container">'
    # Sort bench: GK first, then others
    if not bench.empty:
        bench['pos_order'] = bench['position'].map(pos_order)
        bench = bench.sort_values('pos_order')
        for _, p in bench.iterrows():
            bench_html += _render_player(p)
    bench_html += '</div>'
    
    st.markdown(bench_html, unsafe_allow_html=True)
