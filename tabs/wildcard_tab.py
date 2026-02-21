"""Wildcard Planner Tab - Generate optimal 15-man squad."""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import safe_numeric, round_df, style_df_with_injuries, calculate_consensus_ep, get_consensus_label
from optimizer import MAX_PLAYERS_PER_TEAM, MAX_BUDGET
from components.styles import render_section_title


def generate_wildcard_squad(
    df: pd.DataFrame,
    processor,
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
    
    # Recalculate Poisson EP for THIS horizon (like analytics/ML tabs)
    if horizon > 1:
        try:
            from poisson_ep import calculate_poisson_ep_for_dataframe
            fixtures_df = processor.fixtures_df
            current_gw = processor.fetcher.get_current_gameweek()
            team_stats = st.session_state.get('_understat_team_stats', None)
            poisson_result = calculate_poisson_ep_for_dataframe(
                df, fixtures_df, current_gw, team_stats=team_stats, horizon=horizon
            )
            for col in ('expected_points_poisson', 'games_in_horizon', 'fixture_quality_factor'):
                if col in poisson_result.columns:
                    df[col] = poisson_result[col].values
        except Exception:
            pass
    
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
    
    # Calculate avg per-GW values for display
    if horizon > 1:
        result['avg_poisson_ep'] = (safe_numeric(result.get('poisson_xp_total', result.get('poisson_ep', 0))) / horizon).round(2)
        result['avg_fpl_ep'] = (safe_numeric(result.get('fpl_xp_total', result.get('ep_next_num', 0))) / horizon).round(2)
        result['avg_ml_ep'] = (safe_numeric(result.get('ml_xp_total', result.get('ml_pred', 0))) / horizon).round(2)
        result['avg_consensus_ep'] = (safe_numeric(result['consensus_ep']) / horizon).round(2)
    else:
        result['avg_poisson_ep'] = safe_numeric(result.get('poisson_xp_total', result.get('poisson_ep', 0))).round(2)
        result['avg_fpl_ep'] = safe_numeric(result.get('fpl_xp_total', result.get('ep_next_num', 0))).round(2)
        result['avg_ml_ep'] = safe_numeric(result.get('ml_xp_total', result.get('ml_pred', 0))).round(2)
        result['avg_consensus_ep'] = safe_numeric(result['consensus_ep']).round(2)
    
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
        value=st.session_state.get('wc_last_horizon', 4),
        key="wc_horizon",
        help="Optimize squad for total points over this many gameweeks"
    )
    
    st.markdown("---")
    
    # ── Reactive Generation Logic ──
    # We want to re-run if any input changes OR if the button is clicked.
    
    # 1. Collect current inputs
    current_params = {
        'formation': formation,
        'strategy': strategy,
        'budget': budget,
        'horizon': horizon
    }
    
    # 2. Check if inputs changed
    last_params = st.session_state.get('wc_last_params', {})
    inputs_changed = current_params != last_params
    
    # 3. Handle Generation
    # Triggers: Button click OR Inputs changed (Reactive) OR No squad exists yet
    should_run = st.button("Regenerate Squad", type="secondary", width="stretch") or inputs_changed or 'wc_generated_squad' not in st.session_state
    
    if should_run:
        # If it's an auto-run due to value change, use a lightweight spinner (or none if fast enough)
        # But WC gen can take 1-2s, so spinner is good.
        with st.spinner(f"Optimizing squad for next {horizon} GWs..."):
            squad = generate_wildcard_squad(players_df, processor, formation, strategy, budget, horizon)
            
            if squad.empty:
                st.error("Could not generate a valid squad. Try adjusting budget.")
            else:
                st.session_state['wc_generated_squad'] = squad
                # Update state tracking
                st.session_state['wc_last_params'] = current_params
                st.session_state['wc_last_horizon'] = horizon  # Persist for slider
                
                # If this was a reactive run (not button), we might want to rerun to ensure display updates immediately
                # But treating it as a standard flow usually works in Streamlit.
    
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
            
            # Format display with all 3 models + avg model xP
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']), horizon)
            avg_con_label = f'Avg {con_label}' if horizon > 1 else con_label
            
            display_cols = ['web_name', 'team_name', 'now_cost', 'avg_poisson_ep', 'avg_fpl_ep', 'avg_ml_ep',
                           'avg_consensus_ep', 'form', 'selected_by_percent', 'is_starter']
            display_df = pos_players[[c for c in display_cols if c in pos_players.columns]].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'Avg Poisson', 'Avg FPL', 'Avg ML',
                                 avg_con_label, 'Form', 'EO%', 'Starting']
            display_df['Starting'] = display_df['Starting'].apply(lambda x: 'Starting' if x else 'Bench')
            
            col_config = {
                'Price': st.column_config.NumberColumn(format='£%.1fm'),
                'Avg Poisson': st.column_config.NumberColumn(format='%.2f'),
                'Avg FPL': st.column_config.NumberColumn(format='%.2f'),
                'Avg ML': st.column_config.NumberColumn(format='%.2f'),
                avg_con_label: st.column_config.NumberColumn(format='%.2f'),
                'Form': st.column_config.NumberColumn(format='%.1f'),
                'EO%': st.column_config.NumberColumn(format='%.1f%%'),
            }
            
            st.dataframe(
                style_df_with_injuries(display_df, players_df),
                hide_index=True,
                width="stretch",
                height=min(200, len(pos_players) * 40 + 40),
                column_config=col_config
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
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']), horizon)
            avg_con_label = f'Avg {con_label}' if horizon > 1 else con_label
            
            full_cols = ['web_name', 'team_name', 'position', 'now_cost',
                        'avg_poisson_ep', 'avg_fpl_ep', 'avg_ml_ep',
                        'avg_consensus_ep', 'form', 'selected_by_percent', 'is_starter']
            full_df = squad[[c for c in full_cols if c in squad.columns]].copy()
            full_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Avg Poisson', 'Avg FPL', 'Avg ML',
                              avg_con_label, 'Form', 'EO%', 'Starter']
            full_df = full_df.sort_values(['Starter', 'Pos'], ascending=[False, True])
            full_df['Starter'] = full_df['Starter'].apply(lambda x: 'Yes' if x else 'Bench')
            
            col_config = {
                'Price': st.column_config.NumberColumn(format='£%.1fm'),
                'Avg Poisson': st.column_config.NumberColumn(format='%.2f'),
                'Avg FPL': st.column_config.NumberColumn(format='%.2f'),
                'Avg ML': st.column_config.NumberColumn(format='%.2f'),
                avg_con_label: st.column_config.NumberColumn(format='%.2f'),
                'Form': st.column_config.NumberColumn(format='%.1f'),
                'EO%': st.column_config.NumberColumn(format='%.1f%%'),
            }
            
            st.dataframe(
                style_df_with_injuries(full_df, players_df),
                hide_index=True,
                width="stretch",
                height=600,
                column_config=col_config
            )
