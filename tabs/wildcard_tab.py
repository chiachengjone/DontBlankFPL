"""Wildcard Planner Tab - Generate optimal 15-man squad via ILP.

Migrated from a greedy heuristic to a full Integer Linear Programming
solver (PuLP/CBC) so the engine finds the **globally optimal** 15-man
squad instead of getting trapped in local optima.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np

<<<<<<< Updated upstream
from utils.helpers import safe_numeric, round_df, style_df_with_injuries
=======
from utils.helpers import (
    safe_numeric, round_df, style_df_with_injuries, 
    calculate_consensus_ep, get_consensus_label,
    get_fixture_ease_map, get_opponent_stats_map
)
>>>>>>> Stashed changes
from optimizer import MAX_PLAYERS_PER_TEAM, MAX_BUDGET
from config import (
    SQUAD_SIZE, STARTING_XI, POSITION_CONSTRAINTS,
    BENCH_XP_WEIGHT, RISK_AVERSION_DEFAULT,
)
from components.styles import render_section_title

logger = logging.getLogger(__name__)

# Lazy PuLP import (mirrors optimizer.py)
_pulp_loaded = False

def _ensure_pulp():
    global _pulp_loaded, LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, PULP_CBC_CMD, value
    if not _pulp_loaded:
        from pulp import (
            LpProblem as _P, LpMaximize as _M, LpVariable as _V,
            LpBinary as _B, lpSum as _S, LpStatus as _St,
            PULP_CBC_CMD as _C, value as _val
        )
        LpProblem, LpMaximize, LpVariable, LpBinary = _P, _M, _V, _B
        lpSum, LpStatus, PULP_CBC_CMD, value = _S, _St, _C, _val
        _pulp_loaded = True


def generate_wildcard_squad(
    processor,
    formation: str,
    strategy: str,
    budget: float = 100.0,
    horizon: int = 5,
    risk_aversion: float = RISK_AVERSION_DEFAULT,
    bench_weight: float = BENCH_XP_WEIGHT,
) -> pd.DataFrame:
    """
    Generate globally-optimal 15-man wildcard squad using ILP (PuLP/CBC).

    The solver maximises a weighted objective that includes:
    - Starting XI expected points (primary)
    - Bench expected points scaled by ``bench_weight`` ("safety net")
    - Optional variance penalty (``risk_aversion > 0``)

    Falls back to the legacy greedy algorithm when PuLP is unavailable.
    """
    _ensure_pulp()

    df = processor.players_df.copy()
    fixtures_df = processor.fixtures_df
    current_gw = processor.fetcher.get_current_gameweek()

    # Parse formation
    parts = formation.split('-')
    starting_req = {
        'GKP': 1,
        'DEF': int(parts[0]),
        'MID': int(parts[1]),
        'FWD': int(parts[2])
    }
<<<<<<< Updated upstream
    
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
=======

    # ── 1. Prepare candidate pool ──
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    players = calculate_consensus_ep(df, active_models, horizon=horizon)
    players['now_cost'] = safe_numeric(players['now_cost'], 5)
    players['minutes'] = safe_numeric(players.get('minutes', pd.Series([0]*len(players))))

    # Relaxed reliability filter: elite assets exempt
    xp_threshold = players['consensus_ep'].quantile(0.90)
    players = players[
        (players['minutes'] > 200) | (players['consensus_ep'] >= xp_threshold)
    ].copy()

    # Fixture ease & opponent stats
    ease_map = get_fixture_ease_map(fixtures_df, current_gw, weeks_ahead=horizon)
    players['ease'] = players['team'].map(ease_map).fillna(0.5)
    opp_stats = get_opponent_stats_map(players, fixtures_df, current_gw, weeks_ahead=horizon)
    players['opp_xg'] = players['team'].map(lambda tid: opp_stats.get(tid, {}).get('avg_opp_xg', 1.3))
    players['opp_xgc'] = players['team'].map(lambda tid: opp_stats.get(tid, {}).get('avg_opp_xgc', 1.3))

    # ── 2. Build per-player objective score ──
    def _player_score(row):
        ep = row['consensus_ep']
        ease = row['ease']
        if row['position'] in ['GKP', 'DEF']:
            opp_score = max(3.0 - row['opp_xg'], 0.1) / 3.0
        else:
            opp_score = row['opp_xgc'] / 2.5
        return ep * 0.85 + (ease * 10) * 0.10 + (opp_score * 10) * 0.05

    players['detailed_score'] = players.apply(_player_score, axis=1)
    players['eppm'] = players['detailed_score'] / players['now_cost'].clip(lower=4)

    # Strategy-specific objective value
    if strategy == 'Max Points':
        players['obj'] = players['detailed_score']
>>>>>>> Stashed changes
    elif strategy == 'Value':
        players['obj'] = players['eppm']
    elif strategy == 'Differential':
<<<<<<< Updated upstream
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
=======
        players['selected_by_percent'] = safe_numeric(players['selected_by_percent'])
        players['obj'] = players['detailed_score'] * (1 - players['selected_by_percent'] / 100)
    else:  # Balanced
        players['obj'] = players['detailed_score'] * 0.80 + (players['eppm'] * 5) * 0.20

    # Optional variance penalty (from Monte Carlo std if available)
    if risk_aversion > 0 and 'mc_std' in players.columns:
        players['obj'] = players['obj'] - risk_aversion * safe_numeric(players['mc_std'])
    elif risk_aversion > 0:
        # Heuristic std ≈ 40% of consensus_ep
        players['obj'] = players['obj'] - risk_aversion * players['consensus_ep'] * 0.40

    players = players.reset_index(drop=True)
    pid_list = players['id'].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}

    # ── 3. Build ILP model ──
    prob = LpProblem("WC_Squad_Optimisation", LpMaximize)

    squad_vars = {pid: LpVariable(f"sq_{pid}", cat=LpBinary) for pid in pid_list}
    start_vars = {pid: LpVariable(f"st_{pid}", cat=LpBinary) for pid in pid_list}

    obj_map = players.set_index('id')['obj'].to_dict()

    # Objective: maximise (starter_obj + bench_weight × bench_obj)
    prob += lpSum([
        start_vars[pid] * obj_map[pid]
        + (squad_vars[pid] - start_vars[pid]) * obj_map[pid] * bench_weight
        for pid in pid_list
    ]), "Total_Objective"

    # Budget
    cost_map = players.set_index('id')['now_cost'].to_dict()
    prob += lpSum([squad_vars[pid] * cost_map[pid] for pid in pid_list]) <= budget, "Budget"

    # Squad size = 15
    prob += lpSum([squad_vars[pid] for pid in pid_list]) == SQUAD_SIZE, "Squad_Size"

    # Starting XI = 11
    prob += lpSum([start_vars[pid] for pid in pid_list]) == STARTING_XI, "XI_Size"

    # Start only if in squad
    for pid in pid_list:
        prob += start_vars[pid] <= squad_vars[pid], f"InSquad_{pid}"

    # Max 3 per team
    for team in players['team'].unique():
        team_pids = players[players['team'] == team]['id'].tolist()
        prob += lpSum([squad_vars[pid] for pid in team_pids]) <= MAX_PLAYERS_PER_TEAM, f"Team_{team}"

    # Position constraints (squad totals)
    for pos, lim in POSITION_CONSTRAINTS.items():
        pos_pids = players[players['position'] == pos]['id'].tolist()
        prob += lpSum([squad_vars[pid] for pid in pos_pids]) >= lim['min'], f"SqMin_{pos}"
        prob += lpSum([squad_vars[pid] for pid in pos_pids]) <= lim['max'], f"SqMax_{pos}"

    # Formation constraints (starting XI)
    for pos, req in starting_req.items():
        pos_pids = players[players['position'] == pos]['id'].tolist()
        prob += lpSum([start_vars[pid] for pid in pos_pids]) == req, f"XI_{pos}"

    # ── 4. Solve ──
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    if LpStatus[prob.status] != 'Optimal':
        logger.warning("ILP solver did not find optimal solution (%s); falling back to greedy",
                        LpStatus[prob.status])
        return _greedy_fallback(players, starting_req, budget)

    # ── 5. Extract results ──
    selected_ids = [pid for pid in pid_list if value(squad_vars[pid]) == 1]
    starting_ids = set(pid for pid in pid_list if value(start_vars[pid]) == 1)

    result = players[players['id'].isin(selected_ids)].copy()
    result['is_starter'] = result['id'].isin(starting_ids)

    logger.info("ILP wildcard squad: %d players, cost=%.1f, obj=%.2f",
                len(result), result['now_cost'].sum(), value(prob.objective))
    return result


def _greedy_fallback(players: pd.DataFrame, starting_req: dict, budget: float) -> pd.DataFrame:
    """Legacy greedy algorithm used only when PuLP solver fails."""
    squad_req = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    players = players.sort_values('obj', ascending=False)

    selected, selected_ids, team_counts = [], set(), {}
>>>>>>> Stashed changes
    pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    remaining = budget

    for pos in ['FWD', 'MID', 'DEF', 'GKP']:
        for _, p in players[players['position'] == pos].iterrows():
            if pos_counts[pos] >= starting_req[pos]: break
            if p['id'] in selected_ids: continue
            if team_counts.get(p['team'], 0) >= MAX_PLAYERS_PER_TEAM: continue
            if p['now_cost'] > remaining: continue
            selected.append(p); selected_ids.add(p['id'])
            team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
            pos_counts[pos] += 1; remaining -= p['now_cost']

    for pos in ['DEF', 'MID', 'FWD', 'GKP']:
        for _, p in players[players['position'] == pos].iterrows():
            if pos_counts[pos] >= squad_req[pos]: break
            if p['id'] in selected_ids: continue
            if team_counts.get(p['team'], 0) >= MAX_PLAYERS_PER_TEAM: continue
            if p['now_cost'] > remaining: continue
            selected.append(p); selected_ids.add(p['id'])
            team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
            pos_counts[pos] += 1; remaining -= p['now_cost']

    if not selected:
        return pd.DataFrame()
    result = pd.DataFrame(selected)
    result['is_starter'] = False
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        mask = result['position'] == pos
        result.loc[result[mask].head(starting_req[pos]).index, 'is_starter'] = True
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
        - **Balanced**: Weighs EP and value equally
        - **Max Points**: Selects highest EP players regardless of price
        - **Value**: Prioritizes EPPM (points per million)
        - **Differential**: Low ownership picks for rank gains
        
        **Squad Requirements**
        - 15 players total (2 GK, 5 DEF, 5 MID, 3 FWD)
        - Max 3 players from any single team
        - Must stay within budget (default £100m)
        
<<<<<<< Updated upstream
        **Squad Quality Metrics**
        - Total EP: Combined expected points for the squad
        - Total Value: Squad's EPPM efficiency
        - Ownership Mix: Balance of template vs differential
=======
        **Advanced Selection Logic**
        - **Horizon-Aware**: Considers fixtures and points over your chosen window.
        - **Defensive Solidity**: Defenders and GKPs are scored higher when facing teams with low **Avg Opponent xG**.
        - **Attacking Threat**: Midfielders and Forwards are prioritized when facing teams with high **Avg Opponent xGC** (leaky defenses).
        - **Fixture Ease**: Directly weights the difficulty of upcoming opponents.
>>>>>>> Stashed changes
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
            help="Balanced: EP + Value mix | Max Points: Highest EP | Value: Best EPPM | Differential: Low ownership"
        )
    
    with set_cols[2]:
        budget = st.number_input(
            "Budget (£m)",
            min_value=95.0,
            max_value=110.0,
            value=100.0,
            step=0.1,
            key="wc_budget"
        )
    
    wc_horizon = st.slider("Planning Horizon (GWs)", 1, 8, 5, key="wc_horizon")

    # ── Advanced optimisation settings ──
    with st.expander("Advanced Optimisation Settings"):
        adv_cols = st.columns(2)
        with adv_cols[0]:
            bench_weight = st.slider(
                "Bench Safety-Net Weight",
                min_value=0.0, max_value=0.50, value=float(BENCH_XP_WEIGHT), step=0.05,
                key="wc_bench_weight",
                help="How much the solver values bench xP (0 = ignore bench, 0.5 = weight bench at 50% of starters)"
            )
        with adv_cols[1]:
            risk_aversion = st.slider(
                "Risk Aversion (variance penalty)",
                min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                key="wc_risk_aversion",
                help="0 = max xP (floor bias), higher = penalise volatile picks (ceiling bias when negative is not supported)"
            )

    # ── Automatic Generation ──
    with st.spinner("Optimizing squad with ILP solver..."):
        squad = generate_wildcard_squad(
            processor, formation, strategy, budget,
            horizon=wc_horizon,
            risk_aversion=risk_aversion,
            bench_weight=bench_weight,
        )
        if not squad.empty:
            st.session_state['wc_generated_squad'] = squad
            
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
<<<<<<< Updated upstream
        total_ep = squad['expected_points'].sum()
        starting_ep = squad[squad['is_starter']]['expected_points'].sum()
=======
        starting_df = squad[squad['is_starter']]
        total_ep = safe_numeric(starting_df['consensus_ep']).sum()
        avg_ep_per_match = total_ep / max(wc_horizon, 1)
>>>>>>> Stashed changes
        avg_ownership = squad['selected_by_percent'].mean()
        
        # Calculate League Differential (vs Average Team Template)
        # Template baseline: Avg xP of Top 50 scoring players overall
        active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
        ref_df = calculate_consensus_ep(players_df, active_models, horizon=wc_horizon)
        top_players = ref_df.sort_values('consensus_ep', ascending=False).head(50)
        baseline_avg_xp = (top_players['consensus_ep'].mean() / max(wc_horizon, 1)) * 11
        league_diff = avg_ep_per_match - baseline_avg_xp
        diff_color = "#22c55e" if league_diff >= 0 else "#ef4444"
        diff_prefix = "+" if league_diff >= 0 else ""

        sum_cols = st.columns(4)
        
        with sum_cols[0]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Cost</div>
                <div style="color:#3b82f6;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">£{total_cost:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[1]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg xP / Match</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_ep_per_match:.1f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[2]:
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
<<<<<<< Updated upstream
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Starting XI EP</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{starting_ep:.1f}</div>
=======
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">League Diff</div>
                <div style="color:{diff_color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{diff_prefix}{league_diff:.1f}</div>
>>>>>>> Stashed changes
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
            
<<<<<<< Updated upstream
            # Format display
            display_df = pos_players[['web_name', 'team_name', 'now_cost', 'expected_points', 
                                      'form', 'selected_by_percent', 'is_starter']].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'EP', 'Form', 'EO%', 'Starting']
            display_df['Starting'] = display_df['Starting'].apply(lambda x: 'Starting' if x else 'Bench')
=======
            # Format display (Expanded with Ease and Opponent Stats)
            display_df = pos_players[['web_name', 'team_name', 'now_cost', 'consensus_ep', 
                                      'ease', 'opp_xg', 'opp_xgc', 'is_starter']].copy()
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']))
            xp_col_name = con_label.replace("Avg", "Total")
            
            # Scale ease to 1-5 for display consistency
            display_df['ease'] = (display_df['ease'] * 4 + 1).round(1)
            
            display_df.columns = ['Player', 'Team', 'Price', xp_col_name, 'Ease', 'Opp xG', 'Opp xGC', 'Status']
            display_df['Status'] = display_df['Status'].apply(lambda x: 'Starting' if x else 'Bench')
>>>>>>> Stashed changes
            
            st.dataframe(
                style_df_with_injuries(display_df, players_df, format_dict={
                    'Price': '£{:.1f}m',
<<<<<<< Updated upstream
                    'EP': '{:.2f}',
                    'Form': '{:.1f}',
                    'EO%': '{:.1f}%'
=======
                    con_label: '{:.2f}',
                    'Ease': '{:.1f}',
                    'Opp xG': '{:.2f}',
                    'Opp xGC': '{:.2f}'
>>>>>>> Stashed changes
                }),
                hide_index=True,
                use_container_width=True,
                height=min(250, len(pos_players) * 40 + 40)
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
        with st.expander("View Full Squad Detailed Stats"):
            full_df = squad[['web_name', 'team_name', 'position', 'now_cost', 
<<<<<<< Updated upstream
                            'expected_points', 'form', 'selected_by_percent', 'is_starter']].copy()
            full_df.columns = ['Player', 'Team', 'Pos', 'Price', 'EP', 'Form', 'EO%', 'Starter']
=======
                            'consensus_ep', 'ease', 'opp_xg', 'opp_xgc', 'is_starter']].copy()
            con_label = get_consensus_label(st.session_state.get('active_models', ['ml', 'poisson', 'fpl']))
            xp_col_name = con_label.replace("Avg", "Total")
            
            full_df['ease'] = (full_df['ease'] * 4 + 1).round(1)
            
            full_df.columns = ['Player', 'Team', 'Pos', 'Price', xp_col_name, 'Ease', 'Opp xG', 'Opp xGC', 'Starter']
>>>>>>> Stashed changes
            full_df = full_df.sort_values(['Starter', 'Pos'], ascending=[False, True])
            full_df['Starter'] = full_df['Starter'].apply(lambda x: 'Yes' if x else 'Bench')
            
            st.dataframe(
                style_df_with_injuries(full_df, players_df, format_dict={
                    'Price': '£{:.1f}m',
<<<<<<< Updated upstream
                    'EP': '{:.2f}',
                    'Form': '{:.1f}',
                    'EO%': '{:.1f}%'
=======
                    con_label: '{:.2f}',
                    'Ease': '{:.1f}',
                    'Opp xG': '{:.2f}',
                    'Opp xGC': '{:.2f}'
>>>>>>> Stashed changes
                }),
                hide_index=True,
                use_container_width=True,
                height=600
            )
