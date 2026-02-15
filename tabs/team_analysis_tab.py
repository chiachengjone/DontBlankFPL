"""Team Analysis Tab - Deep-dive into specific teams."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import (
    safe_numeric, style_df_with_injuries,
    calculate_consensus_ep, get_consensus_label
)
from components.styles import render_section_title, render_fixture_badge


def get_fixture_string(processor, team_id: int, num_gws: int = 5) -> list:
    """Get fixture strings for next N gameweeks."""
    try:
        current_gw = processor.fetcher.get_current_gameweek()
        fixtures_df = processor.fixtures_df
        teams_df = processor.teams_df
        team_map = {row['id']: row['short_name'] for _, row in teams_df.iterrows()}
        
        fixtures = fixtures_df[
            (fixtures_df['event'] >= current_gw) &
            (fixtures_df['event'] < current_gw + num_gws) &
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
        ].sort_values('event')
        
        result = []
        for _, fix in fixtures.iterrows():
            is_home = fix['team_h'] == team_id
            opp_id = fix['team_a'] if is_home else fix['team_h']
            opp_name = team_map.get(opp_id, '???')
            venue = 'H' if is_home else 'A'
            fdr = fix.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3)
            result.append({
                'gw': fix['event'],
                'opponent': opp_name,
                'venue': venue,
                'fdr': fdr,
                'display': f"{opp_name}({venue})"
            })
        
        return result
    except Exception:
        return []


def get_predicted_lineup(team_players: pd.DataFrame) -> pd.DataFrame:
    """Select best 11 players by Model xP using standard FPL formation rules."""
    if team_players.empty:
        return team_players
        
    df = team_players.copy()
    # Sort by consensus_ep (Model xP)
    df = df.sort_values('consensus_ep', ascending=False)
    
    lineup = []
    positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    
    # 1. Mandatory Minimums
    # 1 GKP
    gkps = df[df['position'] == 'GKP'].head(1)
    lineup.append(gkps)
    positions['GKP'] += len(gkps)
    
    # 3 DEFs
    defs = df[(df['position'] == 'DEF') & (~df['id'].isin(pd.concat(lineup)['id'] if lineup else []))].head(3)
    lineup.append(defs)
    positions['DEF'] += len(defs)
    
    # 2 MIDs
    mids = df[(df['position'] == 'MID') & (~df['id'].isin(pd.concat(lineup)['id'] if lineup else []))].head(2)
    lineup.append(mids)
    positions['MID'] += len(mids)
    
    # 1 FWD
    fwds = df[(df['position'] == 'FWD') & (~df['id'].isin(pd.concat(lineup)['id'] if lineup else []))].head(1)
    lineup.append(fwds)
    positions['FWD'] += len(fwds)
    
    # 2. Fill best remaining up to 11
    current_ids = pd.concat(lineup)['id'] if lineup else []
    remaining = df[~df['id'].isin(current_ids)]
    
    for _, p in remaining.iterrows():
        if len(pd.concat(lineup)) >= 11:
            break
            
        pos = p['position']
        # Formation limits: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
        max_pos = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        if positions[pos] < max_pos[pos]:
            lineup.append(pd.DataFrame([p]))
            positions[pos] += 1
            
    return pd.concat(lineup) if lineup else pd.DataFrame()


def render_team_analysis_tab(processor, players_df: pd.DataFrame):
    """Team Analysis tab - deep-dive into specific teams."""
    
    st.markdown('<p class="section-title">Team Analysis</p>', unsafe_allow_html=True)
    st.caption("Deep-dive into team assets, fixtures, and stack opportunities")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Team Analysis"):
        st.markdown("""
        **Team Overview**
        - Shows all players from selected Premier League team
        - Useful for identifying stacking opportunities
        
        **Fixture Difficulty Rating (FDR)**
        - 1-2 (Green): Easy fixtures, target these players
        - 3 (Yellow): Medium difficulty
        - 4-5 (Red): Hard fixtures, consider benching/avoiding
        
        **Team Stacking**
        - Owning 2-3 players from same team during good fixtures
        - Maximizes returns when a team has a good run
        - Risk: All players blank if team performs poorly
        
        **Key Assets**
        - Premium: High-priced, high-ceiling players
        - Mid-price: Balanced value options
        - Budget: Cheap enablers to fund premiums
        
        **Clean Sheet Probability**
        - Based on opponent's xG (expected goals) conceded
        - Higher % = better for defenders/goalkeepers
        """)
    
    # Get team list from teams_df
    teams_df = processor.teams_df
    teams = teams_df.to_dict('records')
    team_names = sorted([t['name'] for t in teams])
    team_map = {t['name']: t for t in teams}
    
    # Team selector
    selected_team = st.selectbox("Select Team", team_names, key="team_analysis_select")
    
    if selected_team:
        team_data = team_map[selected_team]
        team_id = team_data['id']
        team_short = team_data['short_name']
        
        # Get team players and calculate consensus
        team_players = players_df[players_df['team'] == team_id].copy()
        active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
        team_players = calculate_consensus_ep(team_players, active_models)
        con_label = get_consensus_label(active_models)
        
        # Use consensus_ep directly - avoid writing to deprecated 'expected_points' column
        team_players['now_cost'] = safe_numeric(team_players['now_cost'], 5)
        team_players['minutes'] = safe_numeric(team_players.get('minutes', pd.Series([0]*len(team_players))))
        team_players['form'] = safe_numeric(team_players.get('form', pd.Series([0]*len(team_players))))
        team_players['selected_by_percent'] = safe_numeric(team_players['selected_by_percent'])
        
        # ── Team Summary ──
        # Calculate stats for ALL teams to determine ranks globally
        all_team_summary = []
        for tid in processor.teams_df['id'].unique():
            t_players = players_df[players_df['team'] == tid].copy()
            # We need consensus_ep for all teams to rank
            t_players = calculate_consensus_ep(t_players, active_models)
            t_lineup = get_predicted_lineup(t_players)
            
            all_team_summary.append({
                'id': tid,
                'lineup_xp': safe_numeric(t_lineup['consensus_ep']).sum(),
                'avg_xp': safe_numeric(t_players['consensus_ep']).mean()
            })
        
        sum_df = pd.DataFrame(all_team_summary)
        
        def get_sum_rank(tid, metric):
            ranks = sum_df[metric].rank(method='min', ascending=False)
            return int(ranks[sum_df['id'] == tid].iloc[0])

        st.markdown(f"### Team Summary (GW{processor.fetcher.get_current_gameweek()} Predicted Lineup)")
        
        sum_cols = st.columns(4)
        
        with sum_cols[0]:
            team_lineup = get_predicted_lineup(team_players)
            lineup_ep = safe_numeric(team_lineup['consensus_ep']).sum()
            cp_rank = get_sum_rank(team_id, 'lineup_xp')
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Best XI {con_label}</div>
                <div style="color:#3b82f6;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{lineup_ep:.2f} <span style="font-size:0.8rem;color:#888;">({cp_rank}º)</span></div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[1]:
            avg_ep = safe_numeric(team_players['consensus_ep']).mean()
            avg_rank = get_sum_rank(team_id, 'avg_xp')
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg Player {con_label}</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_ep:.2f} <span style="font-size:0.8rem;color:#888;">({avg_rank}º)</span></div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[2]:
            top_owned = team_players.nlargest(1, 'selected_by_percent')
            if not top_owned.empty:
                top_name = top_owned.iloc[0]['web_name']
                top_eo = top_owned.iloc[0]['selected_by_percent']
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                    <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Most Owned</div>
                    <div style="color:#f59e0b;font-size:1rem;font-weight:700;">{top_name}</div>
                    <div style="color:#86868b;font-size:0.8rem;">{top_eo:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
        
        with sum_cols[3]:
            num_players = len(team_players)
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Players Available</div>
                <div style="color:#1d1d1f;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{num_players}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Team Attack/Defense Stats (Moved up) ──
        st.markdown("### Team Statistics")
        
        # Calculate stats for ALL teams to determine ranks
        all_team_stats = []
        for tid in processor.teams_df['id'].unique():
            t_players = players_df[players_df['team'] == tid]
            t_attackers = t_players[t_players['position'].isin(['MID', 'FWD'])]
            t_defenders = t_players[t_players['position'].isin(['GKP', 'DEF'])]
            
            all_team_stats.append({
                'id': tid,
                'goals': safe_numeric(t_attackers.get('goals_scored', 0)).sum(),
                'assists': safe_numeric(t_attackers.get('assists', 0)).sum(),
                'xg': safe_numeric(t_attackers.get('us_xG', t_attackers.get('expected_goals', 0))).sum(),
                'xa': safe_numeric(t_attackers.get('us_xA', t_attackers.get('expected_assists', 0))).sum(),
                'cs': safe_numeric(t_defenders.get('clean_sheets', 0)).sum() / max(len(t_defenders), 1),
                'saves': safe_numeric(t_defenders.get('saves', 0)).sum(),
                'cbit': safe_numeric(t_defenders.get('cbit_score', 0)).mean(),
                'gc': safe_numeric(t_defenders.get('goals_conceded', 0)).sum() / max(len(t_defenders), 1)
            })
        
        stats_df = pd.DataFrame(all_team_stats)
        
        # Helper to get rank string
        def get_rank(tid, metric, ascending=False):
            ranks = stats_df[metric].rank(method='min', ascending=ascending)
            rank = int(ranks[stats_df['id'] == tid].iloc[0])
            return f" <span style='font-size:0.75rem;color:#888;'>({rank}º)</span>"

        stat_cols = st.columns(2)
        
        with stat_cols[0]:
            st.markdown("**Attacking Output**")
            attackers = team_players[team_players['position'].isin(['MID', 'FWD'])]
            total_goals = safe_numeric(attackers.get('goals_scored', 0)).sum()
            total_assists = safe_numeric(attackers.get('assists', 0)).sum()
            total_xg = safe_numeric(attackers.get('us_xG', attackers.get('expected_goals', 0))).sum()
            total_xa = safe_numeric(attackers.get('us_xA', attackers.get('expected_assists', 0))).sum()
            
            st.markdown(f'''
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;">
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Goals</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_goals)}{get_rank(team_id, 'goals')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Assists</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_assists)}{get_rank(team_id, 'assists')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">xG</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_xg:.2f}{get_rank(team_id, 'xg')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">xA</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_xa:.2f}{get_rank(team_id, 'xa')}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with stat_cols[1]:
            st.markdown("**Defensive Output**")
            defenders = team_players[team_players['position'].isin(['GKP', 'DEF'])]
            total_cs = safe_numeric(defenders.get('clean_sheets', 0)).sum() / max(len(defenders), 1)
            total_saves = safe_numeric(defenders.get('saves', 0)).sum()
            avg_cbit = safe_numeric(defenders.get('cbit_score', 0)).mean()
            goals_conceded = safe_numeric(defenders.get('goals_conceded', 0)).sum() / max(len(defenders), 1)
            
            st.markdown(f'''
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;">
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Avg CS</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_cs:.2f}{get_rank(team_id, 'cs')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Total Saves</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_saves)}{get_rank(team_id, 'saves')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Avg CBIT</div>
                    <div style="color:#f59e0b;font-weight:600;font-family:'JetBrains Mono',monospace;">{avg_cbit:.2f}{get_rank(team_id, 'cbit')}</div>
                </div>
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.06);padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Avg GC</div>
                    <div style="color:#ef4444;font-weight:600;font-family:'JetBrains Mono',monospace;">{goals_conceded:.2f}{get_rank(team_id, 'gc', ascending=True)}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Fixture Ticker ──
        st.markdown("### Next 8 Fixtures")
        
        fixtures = get_fixture_string(processor, team_id, 8)
        
        if fixtures:
            fix_cols = st.columns(8)
            fdr_colors = {
                1: '#22c55e',  # Very easy
                2: '#4ade80',  # Easy
                3: '#f59e0b',  # Medium
                4: '#ef4444',  # Hard
                5: '#dc2626'   # Very hard
            }
            
            for i, fix in enumerate(fixtures[:8]):
                with fix_cols[i]:
                    color = fdr_colors.get(int(fix['fdr']), '#888')
                    st.markdown(f'''
                    <div style="background:{color};border-radius:8px;padding:0.5rem;text-align:center;">
                        <div style="color:#fff;font-size:0.65rem;font-weight:500;">GW{fix['gw']}</div>
                        <div style="color:#fff;font-size:0.9rem;font-weight:600;">{fix['opponent']}</div>
                        <div style="color:rgba(255,255,255,0.8);font-size:0.65rem;">({fix['venue']})</div>
                    </div>
                    ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Players by Position ──
        st.markdown("### Team Assets")
        
        pos_tabs = st.tabs(["All", "GKP", "DEF", "MID", "FWD"])
        
        for i, pos in enumerate(["All", "GKP", "DEF", "MID", "FWD"]):
            with pos_tabs[i]:
                if pos == "All":
                    display_players = team_players.nlargest(20, 'consensus_ep')
                else:
                    display_players = team_players[team_players['position'] == pos].nlargest(10, 'consensus_ep')
                
                if not display_players.empty:
                    # Add fixture ticker to display
                    display_df = display_players[['web_name', 'position', 'now_cost', 'consensus_ep', 
                                                  'form', 'minutes', 'selected_by_percent']].copy()
                    display_df.columns = ['Player', 'Pos', 'Price', con_label, 'Form', 'Mins', 'EO%']
                    
                    st.dataframe(
                        style_df_with_injuries(display_df, players_df, format_dict={
                            'Price': '£{:.1f}m',
                            con_label: '{:.2f}',
                            'Form': '{:.2f}',
                            'Mins': '{:.0f}',
                            'EO%': '{:.1f}%'
                        }),
                        hide_index=True,
                        use_container_width=True,
                        height=min(400, len(display_df) * 40 + 40)
                    )
                else:
                    st.info(f"No {pos} players available")
        
        st.markdown("---")
        
        # ── Points Distribution (Bubble Chart) ──
        st.markdown("### Point Distribution Cluster")
        st.caption("Player point contributions by position — bubble size reflects total points")
        
        cluster_df = team_players[team_players['total_points'] > 0].copy()
        if not cluster_df.empty:
            import plotly.express as px
            
            # Position order for x-axis
            pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
            cluster_df['pos_val'] = cluster_df['position'].map(pos_order)
            
            fig = px.scatter(
                cluster_df,
                x='position',
                y='total_points',
                size='total_points',
                color='position',
                hover_name='web_name',
                text='web_name',
                category_orders={"position": ["GKP", "DEF", "MID", "FWD"]},
                color_discrete_map={'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(opacity=0.7, line=dict(width=1, color='White')),
                hovertemplate='<b>%{hovertext}</b><br>Points: %{y}<extra></extra>'
            )
            
            fig.update_layout(
                margin=dict(t=30, l=40, r=40, b=40),
                height=450,
                showlegend=False,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', size=12),
                xaxis=dict(title="Position", gridcolor='#f0f0f0'),
                yaxis=dict(title="Total Points", gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True, key="team_point_bubble")
        else:
            st.info("No point data available for this team")
            
        st.markdown("---")
        
        # ── Stack Analysis ──
        st.markdown("### Stack Analysis")
        st.caption("Best combinations of 2-3 players from this team")
        
        # Calculate stack value
        viable_stack = team_players[team_players['minutes'] > 200].nlargest(6, 'consensus_ep')
        
        if len(viable_stack) >= 2:
            stack_options = []
            
            # Double stack (2 players)
            for i, p1 in viable_stack.head(4).iterrows():
                for j, p2 in viable_stack.iterrows():
                    if i < j:
                        combined_ep = safe_numeric(pd.Series([p1['consensus_ep']])).iloc[0] + safe_numeric(pd.Series([p2['consensus_ep']])).iloc[0]
                        combined_cost = p1['now_cost'] + p2['now_cost']
                        stack_options.append({
                            'players': f"{p1['web_name']} + {p2['web_name']}",
                            'positions': f"{p1['position']} + {p2['position']}",
                            'ep': combined_ep,
                            'cost': combined_cost,
                            'value': combined_ep / combined_cost if combined_cost > 0 else 0
                        })
            
            # Sort by EP
            stack_options = sorted(stack_options, key=lambda x: x['ep'], reverse=True)[:5]
            
            if stack_options:
                stack_df = pd.DataFrame(stack_options)
                stack_df.columns = ['Players', 'Positions', 'Combined EP', 'Cost', 'Value']
                
                st.dataframe(
                    stack_df.style.format({
                        'Combined EP': '{:.2f}',
                        'Cost': '£{:.1f}m',
                        'Value': '{:.3f}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("Not enough viable players to analyze stacks")
        
        st.markdown("---")
        
        # ── Rotation Risk Matrix ──
        st.markdown("### Rotation Risk Analysis")
        st.caption("Based on minutes played and chance of playing")
        
        rot_players = team_players[team_players['minutes'] > 0].copy()
        rot_players['mins_per_match'] = rot_players['minutes'] / rot_players['minutes'].max() * 100
        rot_players['cop'] = safe_numeric(rot_players.get('chance_of_playing_next_round', pd.Series([100]*len(rot_players))))
        
        # Classify rotation risk
        def classify_rotation(row):
            if row['cop'] < 50:
                return 'High Risk'
            elif row['mins_per_match'] < 60:
                return 'Rotation Risk'
            elif row['cop'] >= 75 and row['mins_per_match'] >= 80:
                return 'Nailed'
            else:
                return 'Some Risk'
        
        rot_players['risk'] = rot_players.apply(classify_rotation, axis=1)
        
        risk_colors = {
            'Nailed': '#22c55e',
            'Some Risk': '#f59e0b',
            'Rotation Risk': '#ef4444',
            'High Risk': '#dc2626'
        }
        
        risk_cols = st.columns(4)
        for i, risk in enumerate(['Nailed', 'Some Risk', 'Rotation Risk', 'High Risk']):
            with risk_cols[i]:
                risk_players = rot_players[rot_players['risk'] == risk]['web_name'].tolist()[:3]
                color = risk_colors[risk]
                players_str = ', '.join(risk_players) if risk_players else 'None'
                
                st.markdown(f'''
                <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid {color};border-radius:8px;padding:0.75rem;">
                    <div style="color:{color};font-weight:600;font-size:0.85rem;">{risk}</div>
                    <div style="color:#86868b;font-size:0.8rem;margin-top:0.3rem;">{players_str}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
