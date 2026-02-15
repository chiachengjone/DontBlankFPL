"""Optimization tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import safe_numeric, style_df_with_injuries, round_df
from optimizer import MAX_PLAYERS_PER_TEAM
from config import TRANSFER_HIT_COST
from fpl_api import MAX_FREE_TRANSFERS


def render_optimization_tab(processor, players_df: pd.DataFrame, fetcher):
    """Optimization tab - Transfer recommendations based on team."""
    
    st.markdown('<p class="section-title">Transfer Optimizer</p>', unsafe_allow_html=True)
    
    # Metrics explanation dropdown
    with st.expander("Understanding Transfer Optimizer"):
        st.markdown("""
        **Planning Horizon**
        - Number of gameweeks to optimize for (3-10)
        - Longer horizon considers fixture swings
        
        **Free Transfers**
        - How many FTs you have available (1-5 max in 2025/26)
        - Using more than FTs available costs -4 pts per extra transfer
        
        **Strategy Options**
        - **Balanced**: 35% EP, 20% Form, 20% Value, 15% Mins, 10% Diff
        - **Maximum Points**: 55% EP, 25% Form, 20% Mins (best for high budget)
        - **Differential**: 35% EP, 35% Diff, 15% Form, 15% Mins (rank chasing)
        - **Value**: 45% Value (EP/m), 30% EP, 25% Mins (tight budget)
        
        **Transfer Recommendations**
        - Shows best OUT → IN swaps for your squad
        - EP Gain: Expected points improvement
        - Value: Price difference (positive = saves money)
        
        **Hit Analysis**
        - Shows if taking hits (-4, -8, etc.) is worth it
        - Break-even: Points needed to justify the hit
        """)
    
    # Configuration -- Team ID comes from the header input
    team_id = st.session_state.get('fpl_team_id', 0)
    if team_id == 0:
        st.warning("Enter your FPL Team ID in the header bar above to get personalised recommendations.")
    elif team_id < 1 or team_id > 99999999:
        st.error("Invalid Team ID. Must be between 1 and 99999999.")
        team_id = 0
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        weeks_ahead = st.slider(
            "Planning Horizon", min_value=3, max_value=10,
            value=st.session_state.get('pref_weeks_ahead', 5),
            key="opt_horizon",
        )
    with c2:
        from fpl_api import MAX_FREE_TRANSFERS
        free_transfers = st.slider("Free Transfers", min_value=1, max_value=MAX_FREE_TRANSFERS, value=1, key="opt_fts")
    with c3:
        strategies = ['Balanced', 'Maximum Points', 'Differential', 'Value']
        strategy = st.selectbox(
            "Strategy", strategies,
            index=strategies.index(st.session_state.get('pref_strategy', 'Balanced')),
            key="opt_strategy",
        )
    
    bank = st.number_input("Bank (remaining budget)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, key="opt_bank")
    
    if st.button("Analyze Transfers", type="primary", use_container_width=True):
        with st.spinner("Analyzing your squad..."):
            try:
                analyze_transfers(processor, players_df, fetcher, team_id, weeks_ahead, free_transfers, strategy, bank)
            except ValueError as e:
                st.error(f"Team data error: {e}")
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
            has_real_team = True
        elif isinstance(picks_data, list):
            your_squad = [p.get('element', p) for p in picks_data]
            has_real_team = len(your_squad) > 0
        else:
            your_squad = []
            has_real_team = False
            st.warning(f"Team ID {team_id}: No squad data found. Team may be private or ID invalid. Showing general recommendations.")
    except Exception as e:
        your_squad = []
        has_real_team = False
        st.warning(f"Could not fetch team data for ID {team_id}: {str(e)}. Showing general recommendations.")
    
    # Prepare EP metrics - Horizon-Aware Re-calculation
    try:
        from poisson_ep import calculate_poisson_ep_for_dataframe
        fixtures_df = fetcher.get_fixtures_df()
        current_gw = fetcher.get_current_gameweek()
        team_stats = st.session_state.get('_understat_team_stats', None)
        
        # Calculate Poisson EP for the specific horizon chosen in the UI
        # We re-run it here so that the "Poisson EP" column reflects the averaged potential over the whole window
        temp_df = calculate_poisson_ep_for_dataframe(
            featured_df, fixtures_df, current_gw, team_stats=team_stats, horizon=weeks_ahead
        )
        
        # We take the cumulative Poisson for the horizon and divide by weeks to get a per-GW average
        # This keeps the metrics comparable and prevents massive inflation over 5-10 week windows
        featured_df['poisson_ep'] = (safe_numeric(temp_df['expected_points_poisson']) / weeks_ahead).round(2)
    except Exception as e:
        st.error(f"Poisson re-calculation failed: {e}")
        featured_df['poisson_ep'] = safe_numeric(featured_df.get('expected_points_poisson', 2.0))

    featured_df['fpl_ep'] = safe_numeric(featured_df.get('ep_next_num', featured_df.get('ep_next', 2.0)))
    
    # Force Blended EP as primary optimization target (70% Average Poisson, 30% FPL)
    featured_df['blended_ep'] = (featured_df['poisson_ep'] * 0.7 + featured_df['fpl_ep'] * 0.3).round(2)
    ep_col = 'blended_ep'
    
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
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
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
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
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
        out_display = out_candidates[['web_name', 'team_name', 'position', 'now_cost', 'poisson_ep', 'fpl_ep', 'blended_ep', 'form', 'transfer_score']].copy()
        out_display.columns = ['Player', 'Team', 'Pos', 'Price', 'Poisson', 'FPL', 'Blend', 'Form', 'Score']
        st.dataframe(style_df_with_injuries(out_display), hide_index=True, use_container_width=True)
        
        st.markdown("**Recommended IN** (best available)", unsafe_allow_html=True)


def render_ai_transfer_plan(current_squad_df, available_df, ep_col, free_transfers, bank):
    """Render AI-generated transfer plan that decides how many transfers to make.
    
    Evaluates up to free_transfers + 2 extra (hits) and keeps going only when the
    marginal EP gain exceeds the 4-point hit cost.  Always respects FPL constraints:
    - 3 players max per team
    - Position-for-position swaps only
    - Budget constraint
    """
    st.markdown('<p class="section-title">AI Transfer Plan</p>', unsafe_allow_html=True)
    st.caption(
        f"AI evaluates 0-{min(free_transfers + 2, 5)} transfers and recommends the optimal number. "
        f"Free transfers: {free_transfers} | Extra transfers cost {TRANSFER_HIT_COST} pts each."
    )

    squad_value = current_squad_df['now_cost'].sum()
    total_budget = squad_value + bank

    # Evaluate increasing numbers of transfers, stopping when unprofitable
    max_evaluate = min(free_transfers + 2, 5)  # max 5 transfers in one go
    # Sort candidates by weakness
    weakness_sorted = current_squad_df.sort_values('transfer_score', ascending=True)

    best_plan = []
    best_net_value = 0.0  # EP gain minus hits
    explored_plans = []

    squad_teams_base = current_squad_df['team'].value_counts().to_dict()

    for n_transfers in range(1, max_evaluate + 1):
        out_suggestions = weakness_sorted.head(n_transfers)
        transfer_plan = []
        remaining_budget = total_budget
        squad_teams = squad_teams_base.copy()
        used_in_ids = set()
        current_squad_value = squad_value

        for _, out_player in out_suggestions.iterrows():
            pos = out_player['position']
            out_cost = out_player['now_cost']

            pos_avail = available_df[
                (available_df['position'] == pos) &
                (~available_df['id'].isin(used_in_ids))
            ].copy()

            # Team limit check
            out_team = out_player['team']
            def team_ok(row, _out_team=out_team, _squad_teams=squad_teams):
                team = row['team']
                current_count = _squad_teams.get(team, 0)
                if team == _out_team:
                    return current_count - 1 < MAX_PLAYERS_PER_TEAM
                return current_count < MAX_PLAYERS_PER_TEAM

            pos_avail = pos_avail[pos_avail.apply(team_ok, axis=1)]

            max_spend = remaining_budget - (current_squad_value - out_cost)
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
                current_squad_value = current_squad_value - out_cost + best_in['now_cost']

        if len(transfer_plan) < n_transfers:
            break  # couldn't fill all slots, stop exploring

        ep_gain = sum(t['in_ep'] - t['out_ep'] for t in transfer_plan)
        paid_transfers = max(0, n_transfers - free_transfers)
        hit_cost = paid_transfers * TRANSFER_HIT_COST
        net_value = ep_gain - hit_cost

        explored_plans.append({
            'n': n_transfers,
            'plan': transfer_plan,
            'ep_gain': ep_gain,
            'hits': paid_transfers,
            'hit_cost': hit_cost,
            'net_value': net_value,
            'bank_after': bank - sum(t['in_price'] - t['out_price'] for t in transfer_plan),
        })

        if net_value > best_net_value:
            best_net_value = net_value
            best_plan = transfer_plan

    # Show comparison table
    if explored_plans:
        st.markdown("**Transfer Options Compared**")
        comp_rows = []
        best_n = 0
        for ep in explored_plans:
            is_best = (ep['plan'] == best_plan)
            if is_best:
                best_n = ep['n']
            free_label = f"{min(ep['n'], free_transfers)} free"
            hit_label = f" + {ep['hits']} hit" if ep['hits'] > 0 else ""
            comp_rows.append({
                'Transfers': f"{ep['n']} ({free_label}{hit_label})",
                'EP Gain': round(ep['ep_gain'], 1),
                'Hit Cost': f"-{ep['hit_cost']}" if ep['hit_cost'] > 0 else "0",
                'Net Value': round(ep['net_value'], 1),
                'Recommended': ">> YES" if is_best else "",
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Display the best plan
    if best_plan:
        paid = max(0, len(best_plan) - free_transfers)
        hit_total = paid * TRANSFER_HIT_COST
        if paid > 0:
            st.markdown(
                f"**Recommended: {len(best_plan)} transfer(s)** "
                f"({free_transfers} free + {paid} hit = -{hit_total} pts)"
            )
        else:
            st.markdown(f"**Recommended: {len(best_plan)} free transfer(s)**")

        for t in best_plan:
            tc1, tc2, tc3 = st.columns([5, 1, 5])
            with tc1:
                st.markdown(f'''<div class="rule-card">
                    <div style="color:#ef4444;font-weight:600;">OUT: {t['out_name']}</div>
                    <div class="rule-label">{t['out_team']} | {t['out_pos']} | {t['out_price']:.1f}m | Blend {t['out_ep']:.2f}</div>
                </div>''', unsafe_allow_html=True)
            with tc2:
                st.markdown('<div style="text-align:center;padding-top:1rem;color:#fff;font-size:1.5rem;">></div>', unsafe_allow_html=True)
            with tc3:
                st.markdown(f'''<div class="rule-card">
                    <div style="color:#22c55e;font-weight:600;">IN: {t['in_name']}</div>
                    <div class="rule-label">{t['in_team']} | {t['in_pos']} | {t['in_price']:.1f}m | Blend {t['in_ep']:.2f}</div>
                </div>''', unsafe_allow_html=True)

        # Summary
        total_out_cost = sum(t['out_price'] for t in best_plan)
        total_in_cost = sum(t['in_price'] for t in best_plan)
        net_spend = total_in_cost - total_out_cost
        ep_gain = sum(t['in_ep'] - t['out_ep'] for t in best_plan)
        net_val = ep_gain - hit_total

        spend_color = '#ef4444' if net_spend > 0 else '#22c55e'
        ep_color = '#22c55e' if ep_gain > 0 else '#ef4444'
        net_color = '#22c55e' if net_val > 0 else '#ef4444'
        hit_html = (
            f' | <span style="color:#888;">Hit:</span>'
            f' <span style="color:#ef4444;font-weight:600;">-{hit_total} pts</span>'
        ) if hit_total > 0 else ''

        st.markdown(
            f'<div class="rule-card" style="margin-top:0.5rem;">'
            f'<span style="color:#888;">Net Spend:</span> '
            f'<span style="color:{spend_color};font-weight:600;">{net_spend:+.1f}m</span>'
            f' | <span style="color:#888;">Bank After:</span> '
            f'<span style="color:#1d1d1f;font-weight:600;">{bank - net_spend:.1f}m</span>'
            f' | <span style="color:#888;">EP Gain:</span> '
            f'<span style="color:{ep_color};font-weight:600;">{ep_gain:+.1f}</span>'
            f'{hit_html}'
            f' | <span style="color:#888;">Net:</span> '
            f'<span style="color:{net_color};font-weight:600;">{net_val:+.1f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif explored_plans:
        st.info("All transfer options have negative net value. Consider rolling your free transfer.")
    else:
        st.info("Could not generate transfer plan. Budget may be too tight.")


def render_position_recommendations(available_df, ep_col):
    """Render recommendations by position."""
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_df = available_df[available_df['position'] == pos].nlargest(5, 'transfer_score')
        if not pos_df.empty:
            st.markdown(f"**{pos} Recommendations**")
            display_df = pos_df[['web_name', 'team_name', 'now_cost', 'poisson_ep', 'fpl_ep', 'blended_ep', 'form', 'selected_by_percent', 'transfer_score']].copy()
            display_df.columns = ['Player', 'Team', 'Price', 'Poisson', 'FPL', 'Blend', 'Form', 'Owned%', 'Score']
            st.dataframe(style_df_with_injuries(display_df), hide_index=True, use_container_width=True)


def render_top_picks(available_df, ep_col):
    """Render top 10 overall picks."""
    st.markdown('<p class="section-title">Top 10 Overall Picks</p>', unsafe_allow_html=True)
    top_10 = available_df.nlargest(10, 'transfer_score')
    top_display = top_10[['web_name', 'team_name', 'position', 'now_cost', 'poisson_ep', 'fpl_ep', 'blended_ep', 'form', 'selected_by_percent', 'transfer_score']].copy()
    top_display.columns = ['Player', 'Team', 'Pos', 'Price', 'Poisson', 'FPL', 'Blend', 'Form', 'Owned%', 'Score']
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
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_title='Expected Points'
    )
    st.plotly_chart(fig_proj, use_container_width=True, key='opt_points_projection')


