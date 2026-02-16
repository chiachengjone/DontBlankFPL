"""Wildcard Planner Tab - Generate optimal 15-man squad."""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import safe_numeric, round_df, style_df_with_injuries, calculate_consensus_ep, get_consensus_label
from optimizer import MAX_PLAYERS_PER_TEAM, MAX_BUDGET
from components.styles import render_section_title


def generate_wildcard_squad(
    df: pd.DataFrame,
    formation: str,
    strategy: str,
    budget: float = 100.0,
    horizon: int = 4
) -> pd.DataFrame:
    """
    Generate optimal 15-man wildcard squad using a robust greedy algorithm.
    Ensures 15 players are ALWAYS returned if technically possible.
    """
    # Parse formation for starting XI
    parts = formation.split('-')
    starting_req = {'GKP': 1, 'DEF': int(parts[0]), 'MID': int(parts[1]), 'FWD': int(parts[2])}
    squad_req = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    
    # Standardize data
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    players = calculate_consensus_ep(df.copy(), active_models, horizon=horizon)
    players['now_cost'] = safe_numeric(players['now_cost'], 5)
    players['form'] = safe_numeric(players.get('form', pd.Series([0]*len(players))))
    players['selected_by_percent'] = safe_numeric(players['selected_by_percent'])
    players['eppm'] = safe_numeric(players['consensus_ep']) / players['now_cost'].clip(lower=4.0)
    players['minutes'] = safe_numeric(players.get('minutes', pd.Series([0]*len(players))))
    
    # Strategy Scoring
    if strategy == 'Max Points':
        players['score'] = players['consensus_ep']
    elif strategy == 'Value':
        players['score'] = players['eppm']
    elif strategy == 'Differential':
        players['score'] = safe_numeric(players['consensus_ep']) * (1 - players['selected_by_percent'] / 100)
    else: # Balanced
        players['score'] = safe_numeric(players['consensus_ep']) * 0.6 + players['eppm'] * 2 + players['form'] * 0.2

    # Sort by score primarily
    players = players.sort_values('score', ascending=False)

    selected = []
    selected_ids = set()
    team_counts = {}
    remaining_budget = budget

    # Minimum prices for reserve calculation
    MIN_PRICE = {'GKP': 4.0, 'DEF': 4.0, 'MID': 4.4, 'FWD': 4.5}
    
    # ── Iterative Selection ──
    # Build list of 15 slots we need to fill
    slots_to_fill = []
    # Fill starters first (to prioritize best players in starting XI)
    for pos in ['FWD', 'MID', 'DEF', 'GKP']:
        for _ in range(starting_req[pos]):
            slots_to_fill.append(pos)
    # Then fill bench slots
    for pos in ['DEF', 'MID', 'FWD', 'GKP']:
        for _ in range(squad_req[pos] - starting_req[pos]):
            slots_to_fill.append(pos)

    for i, pos in enumerate(slots_to_fill):
        # Calculate budget reserve needed for REMAINING slots (i+1 to 14)
        remaining_slots = slots_to_fill[i+1:]
        reserve_needed = sum(MIN_PRICE[p] for p in remaining_slots)
        
        # Filter valid candidates
        max_allowed_price = remaining_budget - reserve_needed
        
        valid_players = players[
            (players['position'] == pos) & 
            (~players['id'].isin(selected_ids)) &
            (players['now_cost'] <= max_allowed_price)
        ].copy()
        
        # Filter by team limit (max 3)
        valid_players = valid_players[valid_players['team'].apply(lambda t: team_counts.get(t, 0) < MAX_PLAYERS_PER_TEAM)]

        if valid_players.empty:
            # FALLBACK: If no players fit the "smart" budget/team constraint, 
            # pick the absolute cheapest available valid player for this position.
            fallback_players = players[
                (players['position'] == pos) & 
                (~players['id'].isin(selected_ids))
            ].copy()
            fallback_players = fallback_players[fallback_players['team'].apply(lambda t: team_counts.get(t, 0) < MAX_PLAYERS_PER_TEAM)]
            fallback_players = fallback_players.sort_values('now_cost', ascending=True)
            
            if fallback_players.empty:
                # Should be impossible with FPL data
                continue
            
            p = fallback_players.iloc[0]
        else:
            # Pick the best scoring player available within budget reserve constraint
            p = valid_players.iloc[0]
            
        # Add player
        selected.append(p)
        selected_ids.add(p['id'])
        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
        remaining_budget -= p['now_cost']

    if len(selected) < 15:
        return pd.DataFrame()

    result = pd.DataFrame(selected)
    
    # Strictly re-mark starting XI based on formation (top-scoring in each pos)
    result['is_starter'] = False
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_mask = result['position'] == pos
        # We sort by consensus_ep to ensure we actually start the best ones in the final display
        pos_indices = result[pos_mask].sort_values('consensus_ep', ascending=False).index[:starting_req[pos]]
        result.loc[pos_indices, 'is_starter'] = True
    
    return result


def render_wildcard_tab(processor, players_df: pd.DataFrame):
    """Wildcard Planner tab - generate optimal wildcard squad."""
    
    st.markdown('<p class="section-title">Wildcard Planner</p>', unsafe_allow_html=True)
    st.caption("Generate the optimal 15-man squad based on formation and strategy")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Wildcard Planner"):
        st.markdown("""
        **What is a Wildcard?**
        - Chip that allows unlimited free transfers for one gameweek
        - You get 2 wildcards per season (one before GW20, one after)
        - Best used during fixture swings or to fix a failing squad
        
        **Formation Options**
        - Choose your preferred starting XI structure
        - 3-4-3: Attacking formation, 3 forwards
        - 4-4-2: Balanced, good for clean sheet potential
        - 5-4-1: Defensive, maximizes CS points
        
        **Strategy Options**
        - **Balanced**: Weighs xP and value equally
        - **Max Points**: Selects highest xP players regardless of price
        - **Value**: Prioritizes xP/m (points per million)
        - **Differential**: Low ownership picks for rank gains
        
        **Squad Requirements**
        - 15 players total (2 GK, 5 DEF, 5 MID, 3 FWD)
        - Max 3 players from any single team
        - Must stay within budget (default £100m)
        
        **Squad Quality Metrics**
        - Total xP: Combined xP for the squad
        - Total Value: Squad's xP/m efficiency
        - Ownership Mix: Balance of template vs differential
        """)
    
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
            help="Balanced: xP + Value mix | Max Points: Highest xP | Value: Best xP/m | Differential: Low ownership"
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
        
    horizon = st.slider(
        "Planning Horizon (GWs)",
        min_value=1,
        max_value=10,
        value=4,
        key="wc_horizon",
        help="Optimize squad for total points over this many gameweeks"
    )
    
    st.markdown("---")
    
    # ── Generate Button ──
    if st.button("Generate Wildcard Squad", type="primary", use_container_width=True):
        with st.spinner(f"Optimizing squad for next {horizon} GWs..."):
            squad = generate_wildcard_squad(players_df, formation, strategy, budget, horizon)
            
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
        total_ep = safe_numeric(squad['consensus_ep']).sum()
        starting_ep = safe_numeric(squad[squad['is_starter']]['consensus_ep']).sum()
        avg_ep_per_gw = total_ep / max(1, horizon)
        avg_ownership = squad['selected_by_percent'].mean()
        
        sum_cols = st.columns(4)
        
        with sum_cols[0]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Cost</div>
                <div style="color:#3b82f6;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">£{total_cost:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[1]:
            remaining = budget - total_cost
            color = '#22c55e' if remaining >= 0 else '#ef4444'
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">ITB</div>
                <div style="color:{color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">£{remaining:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[2]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Starter xP ({horizon}GW)</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{starting_ep:.1f}</div>
                <div style="color:#86868b;font-size:0.7rem;">{starting_ep/horizon:.1f} / GW</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[3]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg Ownership</div>
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
            display_df = pos_players[['web_name', 'team_name', 'now_cost', 'consensus_ep', 
                                      'form', 'selected_by_percent', 'is_starter']].copy()
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']), horizon)
            display_df.columns = ['Player', 'Team', 'Price', con_label, 'Form', 'EO%', 'Starting']
            display_df['Starting'] = display_df['Starting'].apply(lambda x: 'Starting' if x else 'Bench')
            
            st.dataframe(
                style_df_with_injuries(display_df, players_df, format_dict={
                    'Price': '£{:.1f}m',
                    con_label: '{:.2f}',
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
            st.error(f"Team limit exceeded: {', '.join(over_limit.index.tolist())}")
        else:
            st.success(f"Valid squad: 15 players, £{total_cost:.1f}m, max 3 per team")
        
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
                            'consensus_ep', 'form', 'selected_by_percent', 'is_starter']].copy()
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']), horizon)
            full_df.columns = ['Player', 'Team', 'Pos', 'Price', con_label, 'Form', 'EO%', 'Starter']
            full_df = full_df.sort_values(['Starter', 'Pos'], ascending=[False, True])
            full_df['Starter'] = full_df['Starter'].apply(lambda x: 'Yes' if x else 'Bench')
            
            st.dataframe(
                style_df_with_injuries(full_df, players_df, format_dict={
                    'Price': '£{:.1f}m',
                    con_label: '{:.2f}',
                    'Form': '{:.1f}',
                    'EO%': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True,
                height=600
            )
