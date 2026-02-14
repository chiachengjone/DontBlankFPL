"""Wildcard Planner Tab - Generate optimal 15-man squad."""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import safe_numeric
from optimizer import MAX_PLAYERS_PER_TEAM, MAX_BUDGET
from components.styles import render_section_title


def generate_wildcard_squad(
    df: pd.DataFrame,
    formation: str,
    strategy: str,
    budget: float = 100.0
) -> pd.DataFrame:
    """
    Generate optimal 15-man wildcard squad using greedy algorithm.
    
    FPL Requirements:
    - 15 players total
    - 2 GKP, 5 DEF, 5 MID, 3 FWD
    - Max 3 players per team
    - Budget constraint (default 100m)
    
    Returns DataFrame of selected players.
    """
    # Parse formation for starting XI
    parts = formation.split('-')
    starting_req = {
        'GKP': 1,
        'DEF': int(parts[0]),
        'MID': int(parts[1]),
        'FWD': int(parts[2])
    }
    
    # Full squad requirements (FPL rules)
    squad_req = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    
    # Prepare player data
    players = df.copy()
    players['expected_points'] = safe_numeric(players.get('expected_points', players.get('ep_next', pd.Series([2.0]*len(players)))))
    players['now_cost'] = safe_numeric(players['now_cost'], 5)
    players['form'] = safe_numeric(players.get('form', pd.Series([0]*len(players))))
    players['selected_by_percent'] = safe_numeric(players['selected_by_percent'])
    players['eppm'] = players['expected_points'] / players['now_cost'].clip(lower=4)
    players['minutes'] = safe_numeric(players.get('minutes', pd.Series([0]*len(players))))
    
    # Filter to players with minutes (avoid non-starters)
    players = players[players['minutes'] > 90].copy()
    
    # Sort players based on strategy
    if strategy == 'Max Points':
        players = players.sort_values('expected_points', ascending=False)
    elif strategy == 'Value':
        players = players.sort_values('eppm', ascending=False)
    elif strategy == 'Differential':
        # Low ownership + good EP
        players['diff_score'] = players['expected_points'] * (1 - players['selected_by_percent'] / 100)
        players = players.sort_values('diff_score', ascending=False)
    else:  # Balanced
        players['balanced_score'] = players['expected_points'] * 0.6 + players['eppm'] * 2 + players['form'] * 0.2
        players = players.sort_values('balanced_score', ascending=False)
    
    # Greedy selection
    selected = []
    selected_ids = set()
    team_counts = {}
    pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    remaining_budget = budget
    
    # First pass: Fill starting XI positions with best players
    for pos in ['FWD', 'MID', 'DEF', 'GKP']:  # Fill attacking positions first
        target = starting_req[pos]
        pos_players = players[players['position'] == pos].copy()
        
        for _, p in pos_players.iterrows():
            if pos_counts[pos] >= target:
                break
            
            pid = p['id']
            team = p['team']
            cost = p['now_cost']
            
            # Check constraints
            if pid in selected_ids:
                continue
            if team_counts.get(team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
            if cost > remaining_budget:
                continue
            
            # Add player
            selected.append(p)
            selected_ids.add(pid)
            team_counts[team] = team_counts.get(team, 0) + 1
            pos_counts[pos] += 1
            remaining_budget -= cost
    
    # Second pass: Fill bench positions
    for pos in ['DEF', 'MID', 'FWD', 'GKP']:  # Order for bench importance
        target = squad_req[pos]
        pos_players = players[players['position'] == pos].copy()
        
        for _, p in pos_players.iterrows():
            if pos_counts[pos] >= target:
                break
            
            pid = p['id']
            team = p['team']
            cost = p['now_cost']
            
            # Check constraints
            if pid in selected_ids:
                continue
            if team_counts.get(team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
            if cost > remaining_budget:
                continue
            
            # Add player
            selected.append(p)
            selected_ids.add(pid)
            team_counts[team] = team_counts.get(team, 0) + 1
            pos_counts[pos] += 1
            remaining_budget -= cost
    
    if not selected:
        return pd.DataFrame()
    
    result = pd.DataFrame(selected)
    
    # Mark starting XI
    result['is_starter'] = False
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_mask = result['position'] == pos
        pos_players = result[pos_mask].head(starting_req[pos]).index
        result.loc[pos_players, 'is_starter'] = True
    
    return result


def render_wildcard_tab(processor, players_df: pd.DataFrame):
    """Wildcard Planner tab - generate optimal wildcard squad."""
    
    st.markdown('<p class="section-title">Wildcard Planner</p>', unsafe_allow_html=True)
    st.caption("Generate the optimal 15-man squad based on formation and strategy")
    
    # ── Settings ──
    st.markdown("### Squad Settings")
    
    set_cols = st.columns([1, 1, 1])
    
    with set_cols[0]:
        formation = st.selectbox(
            "Formation",
            ['3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1'],
            key="wc_formation",
            help="Starting XI formation"
        )
    
    with set_cols[1]:
        strategy = st.selectbox(
            "Strategy",
            ['Balanced', 'Max Points', 'Value', 'Differential'],
            key="wc_strategy",
            help="Balanced: EP + Value mix | Max Points: Highest EP | Value: Best EPPM | Differential: Low ownership"
        )
    
    with set_cols[2]:
        budget = st.number_input(
            "Budget (£m)",
            min_value=95.0,
            max_value=105.0,
            value=100.0,
            step=0.5,
            key="wc_budget"
        )
    
    st.markdown("---")
    
    # ── Generate Button ──
    if st.button("🚀 Generate Wildcard Squad", type="primary", use_container_width=True):
        with st.spinner("Optimizing squad..."):
            squad = generate_wildcard_squad(players_df, formation, strategy, budget)
            
            if squad.empty:
                st.error("Could not generate a valid squad. Try adjusting budget.")
            else:
                st.session_state['wc_generated_squad'] = squad
    
    # ── Display Results ──
    if 'wc_generated_squad' in st.session_state:
        squad = st.session_state['wc_generated_squad']
        
        st.markdown("### Generated Squad")
        
        # Summary stats
        total_cost = squad['now_cost'].sum()
        total_ep = squad['expected_points'].sum()
        starting_ep = squad[squad['is_starter']]['expected_points'].sum()
        avg_ownership = squad['selected_by_percent'].mean()
        
        sum_cols = st.columns(4)
        
        with sum_cols[0]:
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Cost</div>
                <div style="color:#3b82f6;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">£{total_cost:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[1]:
            remaining = budget - total_cost
            color = '#22c55e' if remaining >= 0 else '#ef4444'
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">ITB</div>
                <div style="color:{color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">£{remaining:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[2]:
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Starting XI EP</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{starting_ep:.1f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[3]:
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg Ownership</div>
                <div style="color:#f59e0b;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_ownership:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display by position
        pos_order = ['GKP', 'DEF', 'MID', 'FWD']
        pos_names = {'GKP': 'Goalkeepers', 'DEF': 'Defenders', 'MID': 'Midfielders', 'FWD': 'Forwards'}
        
        for pos in pos_order:
            pos_players = squad[squad['position'] == pos].copy()
            if pos_players.empty:
                continue
            
            st.markdown(f"**{pos_names[pos]}**")
            
            # Format display
            display_df = pos_players[['web_name', 'team_name', 'now_cost', 'expected_points', 
                                      'form', 'selected_by_percent', 'is_starter']].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'EP', 'Form', 'EO%', 'Starting']
            display_df['Starting'] = display_df['Starting'].apply(lambda x: '⭐' if x else '🪑')
            
            st.dataframe(
                display_df.style.format({
                    'Price': '£{:.1f}m',
                    'EP': '{:.2f}',
                    'Form': '{:.1f}',
                    'EO%': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True,
                height=min(200, len(pos_players) * 40 + 40)
            )
        
        st.markdown("---")
        
        # Team distribution check
        team_counts = squad.groupby('team_name').size()
        
        if (team_counts > 3).any():
            over_limit = team_counts[team_counts > 3]
            st.error(f"⚠️ Team limit exceeded: {', '.join(over_limit.index.tolist())}")
        else:
            st.success(f"✅ Valid squad: 15 players, £{total_cost:.1f}m, max 3 per team")
        
        # Position verification
        pos_counts = squad.groupby('position').size().to_dict()
        required = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        issues = []
        for pos, req in required.items():
            actual = pos_counts.get(pos, 0)
            if actual != req:
                issues.append(f"{pos}: {actual}/{req}")
        
        if issues:
            st.warning(f"Position requirements not met: {', '.join(issues)}")
        
        # Full squad table
        with st.expander("View Full Squad Table"):
            full_df = squad[['web_name', 'team_name', 'position', 'now_cost', 
                            'expected_points', 'form', 'selected_by_percent', 'is_starter']].copy()
            full_df.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Form', 'EO%', 'Starter']
            full_df = full_df.sort_values(['Starter', 'Pos'], ascending=[False, True])
            full_df['Starter'] = full_df['Starter'].apply(lambda x: 'Yes' if x else 'Bench')
            
            st.dataframe(
                full_df.style.format({
                    'Price': '£{:.1f}m',
                    'EP': '{:.2f}',
                    'Form': '{:.1f}',
                    'EO%': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True,
                height=600
            )
