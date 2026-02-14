"""Team Analysis Tab - Deep-dive into specific teams."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric, style_df_with_injuries
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


def render_team_analysis_tab(processor, players_df: pd.DataFrame):
    """Team Analysis tab - deep-dive into specific teams."""
    
    st.markdown('<p class="section-title">Team Analysis</p>', unsafe_allow_html=True)
    st.caption("Deep-dive into team assets, fixtures, and stack opportunities")
    
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
        
        # Get team players
        team_players = players_df[players_df['team'] == team_id].copy()
        team_players['expected_points'] = safe_numeric(team_players.get('expected_points', team_players.get('ep_next', pd.Series([0]*len(team_players)))))
        team_players['now_cost'] = safe_numeric(team_players['now_cost'], 5)
        team_players['minutes'] = safe_numeric(team_players.get('minutes', pd.Series([0]*len(team_players))))
        team_players['form'] = safe_numeric(team_players.get('form', pd.Series([0]*len(team_players))))
        team_players['selected_by_percent'] = safe_numeric(team_players['selected_by_percent'])
        
        # ── Team Summary ──
        st.markdown("### Team Summary")
        
        sum_cols = st.columns(4)
        
        with sum_cols[0]:
            total_ep = team_players['expected_points'].sum()
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Team EP</div>
                <div style="color:#3b82f6;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{total_ep:.1f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[1]:
            avg_ep = team_players['expected_points'].mean()
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg Player EP</div>
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_ep:.2f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with sum_cols[2]:
            top_owned = team_players.nlargest(1, 'selected_by_percent')
            if not top_owned.empty:
                top_name = top_owned.iloc[0]['web_name']
                top_eo = top_owned.iloc[0]['selected_by_percent']
                st.markdown(f'''
                <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Most Owned</div>
                    <div style="color:#f59e0b;font-size:1rem;font-weight:700;">{top_name}</div>
                    <div style="color:#6b6b6b;font-size:0.8rem;">{top_eo:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
        
        with sum_cols[3]:
            num_players = len(team_players)
            st.markdown(f'''
            <div style="background:#141416;border:1px solid #2a2a2e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#6b6b6b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Players Available</div>
                <div style="color:#e8e8e8;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{num_players}</div>
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
                    display_players = team_players.nlargest(20, 'expected_points')
                else:
                    display_players = team_players[team_players['position'] == pos].nlargest(10, 'expected_points')
                
                if not display_players.empty:
                    # Add fixture ticker to display
                    display_df = display_players[['web_name', 'position', 'now_cost', 'expected_points', 
                                                  'form', 'minutes', 'selected_by_percent']].copy()
                    display_df.columns = ['Player', 'Pos', 'Price', 'EP', 'Form', 'Mins', 'EO%']
                    
                    st.dataframe(
                        display_df.style.format({
                            'Price': '£{:.1f}m',
                            'EP': '{:.2f}',
                            'Form': '{:.1f}',
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
        
        # ── Stack Analysis ──
        st.markdown("### Stack Analysis")
        st.caption("Best combinations of 2-3 players from this team")
        
        # Calculate stack value
        viable_stack = team_players[team_players['minutes'] > 200].nlargest(6, 'expected_points')
        
        if len(viable_stack) >= 2:
            stack_options = []
            
            # Double stack (2 players)
            for i, p1 in viable_stack.head(4).iterrows():
                for j, p2 in viable_stack.iterrows():
                    if i < j:
                        combined_ep = p1['expected_points'] + p2['expected_points']
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
                <div style="background:#141416;border:1px solid {color};border-radius:8px;padding:0.75rem;">
                    <div style="color:{color};font-weight:600;font-size:0.85rem;">{risk}</div>
                    <div style="color:#6b6b6b;font-size:0.8rem;margin-top:0.3rem;">{players_str}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Team Attack/Defense Stats ──
        st.markdown("### Team Statistics")
        
        stat_cols = st.columns(2)
        
        with stat_cols[0]:
            st.markdown("**Attacking Output**")
            # Sum attacking stats from team players
            attackers = team_players[team_players['position'].isin(['MID', 'FWD'])]
            total_goals = safe_numeric(attackers.get('goals_scored', pd.Series([0]*len(attackers)))).sum()
            total_assists = safe_numeric(attackers.get('assists', pd.Series([0]*len(attackers)))).sum()
            total_xg = safe_numeric(attackers.get('us_xG', attackers.get('expected_goals', pd.Series([0]*len(attackers))))).sum()
            total_xa = safe_numeric(attackers.get('us_xA', attackers.get('expected_assists', pd.Series([0]*len(attackers))))).sum()
            
            st.markdown(f'''
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;">
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Goals</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_goals)}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Assists</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_assists)}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">xG</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_xg:.1f}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">xA</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_xa:.1f}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with stat_cols[1]:
            st.markdown("**Defensive Output**")
            defenders = team_players[team_players['position'].isin(['GKP', 'DEF'])]
            total_cs = safe_numeric(defenders.get('clean_sheets', pd.Series([0]*len(defenders)))).sum() / max(len(defenders), 1)
            total_saves = safe_numeric(defenders.get('saves', pd.Series([0]*len(defenders)))).sum()
            avg_cbit = safe_numeric(defenders.get('cbit_score', pd.Series([0]*len(defenders)))).mean()
            goals_conceded = safe_numeric(defenders.get('goals_conceded', pd.Series([0]*len(defenders)))).sum() / max(len(defenders), 1)
            
            st.markdown(f'''
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;">
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Avg CS</div>
                    <div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{total_cs:.1f}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Total Saves</div>
                    <div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{int(total_saves)}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Avg CBIT</div>
                    <div style="color:#f59e0b;font-weight:600;font-family:'JetBrains Mono',monospace;">{avg_cbit:.2f}</div>
                </div>
                <div style="background:#141416;border:1px solid #2a2a2e;padding:0.75rem;border-radius:8px;text-align:center;">
                    <div style="color:#6b6b6b;font-size:0.7rem;text-transform:uppercase;">Avg GC</div>
                    <div style="color:#ef4444;font-weight:600;font-family:'JetBrains Mono',monospace;">{goals_conceded:.1f}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
