"""Strategy tab for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

<<<<<<< Updated upstream
from utils.helpers import safe_numeric, get_injury_status, style_df_with_injuries, round_df, normalize_name
from components.charts import create_ep_ownership_scatter
=======
from utils.helpers import (
    safe_numeric, get_injury_status, style_df_with_injuries, round_df, normalize_name,
    calculate_consensus_ep, get_consensus_label, calculate_enhanced_captain_score
)
from components.charts import create_dynamic_player_scatter
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
from components.cards import render_player_detail_card
from fpl_api import CBIT_BONUS_THRESHOLD, CBIT_BONUS_POINTS, MAX_FREE_TRANSFERS, CAPTAIN_MULTIPLIER


def render_strategy_tab(processor, players_df: pd.DataFrame):
    """Strategy tab - EP vs Ownership visualization with filters."""
    
    # Metrics explanation dropdown
    with st.expander("Understanding Strategy Metrics"):
        st.markdown("""
        **2025/26 Season Rules**
        - **5 Max Free Transfers**: Can now bank up to 5 FTs (was 2)
        - **1.25x Captain**: Captain earns 1.25x points (was 2x)
        - **CBIT Bonus**: +2 pts for 10+ in-play actions
        
        **Player Landscape Scatter Plot**
        - X-axis: Ownership % (how many managers own them)
        - Y-axis: Expected Points (predicted GW score)
        - Bubble size: Price (bigger = more expensive)
        
        **Quadrant Strategy**
        - Top-left (high EP, low ownership): Prime differentials
        - Top-right (high EP, high ownership): Essential template picks
        - Bottom-left: Avoid zone
        - Bottom-right: Traps (popular but low EP)
        
        **Color Coding**
        - Green: Attackers (FWD)
        - Blue: Midfielders (MID)
        - Orange: Defenders (DEF)
        - Purple: Goalkeepers (GKP)
        """)
    
    # Season rules at top
    st.markdown('<p class="section-title">2025/26 Season Rules</p>', unsafe_allow_html=True)
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">{MAX_FREE_TRANSFERS}</div>
            <div class="rule-label">Max Free Transfers</div>
        </div>
        ''', unsafe_allow_html=True)
    with r2:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">{CAPTAIN_MULTIPLIER}x</div>
            <div class="rule-label">Captain Multiplier</div>
        </div>
        ''', unsafe_allow_html=True)
    with r3:
        st.markdown(f'''
        <div class="rule-card">
            <div class="rule-value">+{CBIT_BONUS_POINTS}</div>
            <div class="rule-label">CBIT Bonus ({CBIT_BONUS_THRESHOLD} actions)</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Player Landscape</p>', unsafe_allow_html=True)
    
    # Filters
    f1, f2, f3, f4 = st.columns([1, 1, 1.5, 1.5])
    
    with f1:
        pos_filter = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="strat_pos")
    with f2:
        all_teams = sorted(players_df['team_name'].dropna().unique().tolist()) if 'team_name' in players_df.columns else []
        team_filter = st.selectbox("Team", ['All'] + all_teams, key="strat_team")
    with f3:
        max_price = st.slider("Max Price", 4.0, 15.0, 15.0, 0.5, key="strat_price")
    with f4:
        search_player = st.text_input("Search player", placeholder="Type name to highlight...", key="strat_search")
    
    # ── Map/Exploration Controls ──
    st.markdown('<p class="section-title">Player Exploration</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        x_axis = st.selectbox(
            "X-Axis Variable", 
            ["Ownership %", "Form", "Price", "Total Points", "ICT Index", "Value (xP/£)"],
            index=0,
            key="strat_x_axis"
        )
        st.caption("Change the x-axis to explore different player metrics vs Model xP.")
    
    # Use base player data for fast graph rendering
    df = players_df.copy()
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df = df[df['now_cost'] <= max_price]
    
    if pos_filter != 'All':
        df = df[df['position'] == pos_filter]
    
    if team_filter != 'All' and 'team_name' in df.columns:
        df = df[df['team_name'] == team_filter]
    
    # Player detail view when searching
    if search_player and search_player.strip():
        q = normalize_name(search_player.lower().strip())
        
        # Ensure normalized components exist for filtering
        if 'first_normalized' not in players_df.columns:
            players_df['first_normalized'] = players_df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
            players_df['second_normalized'] = players_df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
            
        matched = players_df[
            (players_df['first_normalized'].str.startswith(q, na=False)) |
            (players_df['second_normalized'].str.startswith(q, na=False))
        ].sort_values('selected_by_percent', ascending=False)
        
        if not matched.empty:
            selected = matched.iloc[0]
            
            st.markdown('<p class="section-title">Player Details</p>', unsafe_allow_html=True)
            render_player_detail_card(selected, processor, players_df)
            
            if len(matched) > 1:
                st.caption(f"Other matches: {', '.join(matched['web_name'].tolist()[:10])}")
    
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    # Scatter plot
    if 'expected_points' in df.columns or 'ep_next' in df.columns:
        fig = create_ep_ownership_scatter(df, pos_filter, search_player=search_player)
=======
=======
>>>>>>> Stashed changes
    # Dynamic Scatter plot
    if 'consensus_ep' in df.columns or 'expected_points_poisson' in df.columns or 'ep_next' in df.columns:
        fig = create_dynamic_player_scatter(
            df, 
            x_axis_col=x_axis,
            position_filter=pos_filter, 
            search_player=search_player, 
            ep_label=con_label
        )
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        if fig:
            st.plotly_chart(fig, use_container_width=True, key='strategy_dynamic_scatter')
    else:
        st.info("Data loading...")
    
    # Quick stats
    st.markdown('<p class="section-title">Quick Stats</p>', unsafe_allow_html=True)
    render_quick_stats(df)
    
    # Captain planning
    render_captain_planning(players_df, processor)
    
    # Fixture difficulty
    render_fixture_difficulty(processor)
    
    # Price watch
    render_price_watch(players_df)
    
    # Injury alerts
    render_injury_alerts(players_df)
    
    # Ownership trends
    render_ownership_trends(players_df)


def render_quick_stats(df: pd.DataFrame):
    """Render quick stats row."""
    stat1, stat2, stat3, stat4 = st.columns(4)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    with stat1:
        top_owned = df.nlargest(1, 'selected_by_percent')
        if not top_owned.empty:
            st.metric("Most Owned", top_owned.iloc[0]['web_name'], f"{top_owned.iloc[0]['selected_by_percent']:.1f}%")
    
    with stat2:
        ep_col = 'expected_points' if 'expected_points' in df.columns else 'ep_next'
        if ep_col in df.columns:
            df[ep_col] = safe_numeric(df[ep_col])
            top_ep = df.nlargest(1, ep_col)
            if not top_ep.empty:
                st.metric("Top EP", top_ep.iloc[0]['web_name'], f"{top_ep.iloc[0][ep_col]:.1f}")
    
    with stat3:
        st.metric("Players Shown", len(df))
    
    with stat4:
        avg_price = df['now_cost'].mean()
        st.metric("Avg Price", f"{avg_price:.1f}m")


<<<<<<< Updated upstream
<<<<<<< Updated upstream
def render_form_vs_ep_chart(df: pd.DataFrame):
    """Render Form vs EP bubble chart — size = price, color = position."""
    st.markdown('<p class="section-title">Form vs Expected Points</p>', unsafe_allow_html=True)
    st.caption("Bubble size = price. Top-right = hot and high-ceiling players.")
    
    chart_df = df.copy()
    chart_df['form'] = safe_numeric(chart_df.get('form', pd.Series([0]*len(chart_df))))
    chart_df['ep'] = safe_numeric(chart_df.get('expected_points', chart_df.get('ep_next', pd.Series([2]*len(chart_df)))))
    chart_df['minutes'] = safe_numeric(chart_df.get('minutes', pd.Series([0]*len(chart_df))))
    chart_df = chart_df[chart_df['minutes'] > 200]
    
    if chart_df.empty or len(chart_df) < 5:
        return
    
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    fig = go.Figure()
    for pos, color in pos_colors.items():
        pos_df = chart_df[chart_df['position'] == pos]
        if pos_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pos_df['form'],
            y=pos_df['ep'],
            mode='markers',
            name=pos,
            marker=dict(
                size=pos_df['now_cost'].clip(lower=4, upper=14) * 2.5,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='rgba(0,0,0,0.06)')
            ),
            text=pos_df['web_name'],
            hovertemplate='<b>%{text}</b><br>Form: %{x:.1f}<br>EP: %{y:.1f}<br><extra></extra>'
        ))
    
    # Add quadrant lines at medians
    med_form = chart_df['form'].median()
    med_ep = chart_df['ep'].median()
    
    fig.add_hline(y=med_ep, line_dash="dot", line_color="rgba(0,0,0,0.08)")
    fig.add_vline(x=med_form, line_dash="dot", line_color="rgba(0,0,0,0.08)")
    
    fig.update_layout(
        height=420,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        xaxis=dict(title='Form', gridcolor='#e5e5ea', zerolinecolor='#e5e5ea'),
        yaxis=dict(title='Expected Points', gridcolor='#e5e5ea', zerolinecolor='#e5e5ea'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=30, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='strategy_form_vs_ep')
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes


def render_captain_planning(players_df: pd.DataFrame, processor):
    """Render captain planning section."""
    st.markdown('<p class="section-title">Captain Picks</p>', unsafe_allow_html=True)
    st.caption("Top captain options based on EP, form, and fixture difficulty")
    
    cap_df = players_df.copy()
    cap_df['ep'] = safe_numeric(cap_df.get('expected_points', cap_df.get('ep_next', pd.Series([0]*len(cap_df)))))
    cap_df['form'] = safe_numeric(cap_df.get('form', pd.Series([0]*len(cap_df))))
    cap_df['selected_by_percent'] = safe_numeric(cap_df['selected_by_percent'])
    cap_df['minutes'] = safe_numeric(cap_df.get('minutes', pd.Series([0]*len(cap_df))))
    cap_df = cap_df[cap_df['minutes'] > 500]
    
    cap_df['captain_score'] = cap_df['ep'] * CAPTAIN_MULTIPLIER + cap_df['form'] * 0.3
    cap_df['captain_ev'] = cap_df['ep'] * CAPTAIN_MULTIPLIER
    
    cap_cols = st.columns(5)
    top_caps = cap_df.nlargest(5, 'captain_score')
    
    for i, (_, cap) in enumerate(top_caps.iterrows()):
        with cap_cols[i]:
            injury = get_injury_status(cap)
            injury_badge = f'<span style="color:{injury["color"]}"> [{injury["icon"]}]</span>' if injury['icon'] else ''
            st.markdown(f'''
            <div class="rule-card">
                <div style="font-size:0.75rem;color:#888;">#{i+1}</div>
                <div style="font-size:1.1rem;font-weight:600;color:#fff;">{cap["web_name"]}{injury_badge}</div>
                <div style="color:#888;font-size:0.85rem;">{cap.get("team_name", "")} | {cap["now_cost"]:.1f}m</div>
                <div style="color:#ef4444;font-weight:600;margin-top:0.5rem;">{cap["captain_ev"]:.1f} EV</div>
                <div style="color:#888;font-size:0.75rem;">Form: {cap["form"]:.1f} | Own: {cap["selected_by_percent"]:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)


def render_fixture_difficulty(processor):
    """Render team fixture difficulty ranking with team filtering."""
    st.markdown('<p class="section-title">Team Fixture Difficulty</p>', unsafe_allow_html=True)
    st.caption("Rank teams by schedule difficulty over upcoming gameweeks")
    
    try:
        current_gw = processor.fetcher.get_current_gameweek()
        fixtures = processor.fixtures_df
        teams_df = processor.teams_df
        
        if fixtures is None or fixtures.empty or teams_df is None:
            st.info("Fixture data unavailable")
            return
        
        fixtures = fixtures.copy()
        fixtures['event'] = safe_numeric(fixtures.get('event', pd.Series([0]*len(fixtures))))
        upcoming = fixtures[fixtures['event'] > current_gw].copy()
        
        # Configuration
        cfg1, cfg2, cfg3 = st.columns([1, 1, 1])
        
        with cfg1:
            gw_horizon = st.slider("GW Lookahead", min_value=3, max_value=10, value=6, key="strat_fdr_horizon")
        
        with cfg2:
            team_names_list = ['All Teams'] + sorted([t['name'] for _, t in teams_df.iterrows()])
            selected_team = st.selectbox("Filter Team", team_names_list, key="strat_fdr_team_filter")
        
        with cfg3:
            sort_order = st.selectbox("Sort Order", ["Easiest First", "Hardest First"], key="strat_fdr_sort")
        
        # Calculate FDR for each team over the horizon
        gw_range = list(range(current_gw + 1, min(current_gw + gw_horizon + 1, 39)))
        team_fdr_data = []
        
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_name = team['name']
            team_short = team['short_name']
            
            fixture_details = []
            total_fdr = 0
            game_count = 0
            
            for gw in gw_range:
                gw_fixtures = upcoming[upcoming['event'] == gw]
                
                # Home fixtures
                home_matches = gw_fixtures[gw_fixtures['team_h'] == team_id]
                for _, match in home_matches.iterrows():
                    opponent_id = match['team_a']
                    opponent = teams_df[teams_df['id'] == opponent_id]['short_name'].values
                    opponent = opponent[0] if len(opponent) > 0 else 'UNK'
                    fdr = safe_numeric(pd.Series([match.get('team_h_difficulty', 3)]))[0]
                    fixture_details.append({
                        'GW': gw,
                        'Opponent': f"{opponent} (H)",
                        'FDR': int(fdr),
                        'Home': True
                    })
                    total_fdr += fdr
                    game_count += 1
                
                # Away fixtures
                away_matches = gw_fixtures[gw_fixtures['team_a'] == team_id]
                for _, match in away_matches.iterrows():
                    opponent_id = match['team_h']
                    opponent = teams_df[teams_df['id'] == opponent_id]['short_name'].values
                    opponent = opponent[0] if len(opponent) > 0 else 'UNK'
                    fdr = safe_numeric(pd.Series([match.get('team_a_difficulty', 3)]))[0]
                    fixture_details.append({
                        'GW': gw,
                        'Opponent': f"{opponent} (A)",
                        'FDR': int(fdr),
                        'Home': False
                    })
                    total_fdr += fdr
                    game_count += 1
            
            avg_fdr = total_fdr / game_count if game_count > 0 else 5.0
            
            team_fdr_data.append({
                'team_id': team_id,
                'team_name': team_name,
                'team_short': team_short,
                'avg_fdr': avg_fdr,
                'total_fdr': total_fdr,
                'games': game_count,
                'fixture_details': fixture_details
            })
        
        # Sort by average FDR
        team_fdr_data.sort(key=lambda x: x['avg_fdr'], reverse=(sort_order == "Hardest First"))
        
        if selected_team == 'All Teams':
            # Show ranking table with heatmap
            ranking_data = []
            for rank, team_data in enumerate(team_fdr_data, 1):
                fixtures_str = ', '.join([f["Opponent"] for f in team_data['fixture_details'][:gw_horizon]])
                
                avg = team_data['avg_fdr']
                if avg <= 2.5:
                    difficulty = "Easy"
                elif avg <= 3.0:
                    difficulty = "Medium"
                elif avg <= 3.5:
                    difficulty = "Tough"
                else:
                    difficulty = "Hard"
                
                ranking_data.append({
                    'Rank': rank,
                    'Team': team_data['team_short'],
                    'Avg FDR': round(avg, 2),
                    'Difficulty': difficulty,
                    'Games': team_data['games'],
                    'Next Fixtures': fixtures_str[:60] + ('...' if len(fixtures_str) > 60 else '')
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, hide_index=True, use_container_width=True)
            
            # Heatmap view
            fdr_cols = [f'GW{gw}' for gw in gw_range]
            heatmap_data = []
            for team_data in team_fdr_data:
                row = {'Team': team_data['team_short']}
                for gw in gw_range:
                    gw_fixtures = [f for f in team_data['fixture_details'] if f['GW'] == gw]
                    if gw_fixtures:
                        row[f'GW{gw}'] = int(np.mean([f['FDR'] for f in gw_fixtures]))
                    else:
                        row[f'GW{gw}'] = 3
                heatmap_data.append(row)
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_df[fdr_cols].values,
                x=fdr_cols,
                y=heatmap_df['Team'],
                colorscale=[[0, '#22c55e'], [0.25, '#77c45e'], [0.5, '#f59e0b'], [0.75, '#ef6b4e'], [1, '#ef4444']],
                showscale=True,
                colorbar=dict(title='FDR', tickvals=[1, 2, 3, 4, 5]),
                zmin=1, zmax=5,
                text=heatmap_df[fdr_cols].values,
                texttemplate='%{text}',
                textfont=dict(size=10, color='white'),
                hovertemplate='%{y} %{x}: FDR %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                height=600,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(family='Inter, sans-serif', color='#86868b', size=11),
                margin=dict(l=80, r=40, t=20, b=40),
                yaxis=dict(autorange='reversed' if sort_order == "Easiest First" else True)
            )
            st.plotly_chart(fig, use_container_width=True, key='strategy_fixture_heatmap')
        
        else:
            # Show detailed view for selected team
            team_data = next((t for t in team_fdr_data if t['team_name'] == selected_team), None)
            
            if team_data:
                st.markdown(f"### {selected_team} Fixture Analysis")
                
                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.metric("Average FDR", f"{team_data['avg_fdr']:.2f}")
                with m2:
                    st.metric("Total Games", team_data['games'])
                with m3:
                    # Find rank
                    sorted_by_fdr = sorted(team_fdr_data, key=lambda x: x['avg_fdr'])
                    rank = next((i+1 for i, t in enumerate(sorted_by_fdr) if t['team_name'] == selected_team), 0)
                    st.metric("Schedule Rank", f"{rank}/20", help="1 = easiest schedule")
                with m4:
                    easy_games = len([f for f in team_data['fixture_details'] if f['FDR'] <= 2])
                    st.metric("Easy Fixtures (FDR ≤2)", easy_games)
                
                # Fixture table
                st.markdown("**Upcoming Fixtures**")
                if team_data['fixture_details']:
                    fixture_df = pd.DataFrame(team_data['fixture_details'])
                    
                    def fdr_color(val):
                        if val <= 2:
                            return 'Easy'
                        elif val == 3:
                            return 'Med'
                        elif val == 4:
                            return 'Tough'
                        else:
                            return 'Hard'
                    
                    fixture_df['Difficulty'] = fixture_df['FDR'].apply(fdr_color)
                    fixture_df['Venue'] = fixture_df['Home'].apply(lambda x: 'Home' if x else 'Away')
                    fixture_df = fixture_df[['GW', 'Opponent', 'Venue', 'FDR', 'Difficulty']]
                    
                    st.dataframe(fixture_df, hide_index=True, use_container_width=True)
                    
                    # FDR trend chart
                    gw_fdr = {}
                    for f in team_data['fixture_details']:
                        gw = f['GW']
                        if gw not in gw_fdr:
                            gw_fdr[gw] = []
                        gw_fdr[gw].append(f['FDR'])
                    
                    gws = sorted(gw_fdr.keys())
                    avg_fdrs = [np.mean(gw_fdr[gw]) for gw in gws]
                    
                    colors = ['#22c55e' if fdr <= 2 else '#eab308' if fdr <= 3 
                              else '#f97316' if fdr <= 4 else '#ef4444' for fdr in avg_fdrs]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f'GW{gw}' for gw in gws],
                        y=avg_fdrs,
                        marker_color=colors,
                        text=[f'{fdr:.1f}' for fdr in avg_fdrs],
                        textposition='outside',
                        hovertemplate='GW%{x}: FDR %{y:.1f}<extra></extra>'
                    ))
                    
                    fig.add_hline(y=3.0, line_dash='dash', line_color='#86868b', 
                                  annotation_text='Average (3.0)')
                    
                    fig.update_layout(
                        height=300,
                        template='plotly_white',
                        paper_bgcolor='#ffffff',
                        plot_bgcolor='#ffffff',
                        font=dict(family='Inter, sans-serif', color='#86868b', size=10),
                        margin=dict(l=40, r=20, t=20, b=40),
                        yaxis=dict(title='FDR', range=[0, 5.5]),
                        xaxis=dict(title='')
                    )
                    st.plotly_chart(fig, use_container_width=True, key='team_fdr_trend_strategy')
                    
                    # Key players from this team
                    st.markdown("**Key Players**")
                    team_players = st.session_state.get('players_df', pd.DataFrame())
                    if not team_players.empty and 'team_name' in team_players.columns:
                        team_squad = team_players[team_players['team_name'] == selected_team].copy()
                        if not team_squad.empty:
                            team_squad['ep'] = safe_numeric(team_squad.get('expected_points', team_squad.get('ep_next', pd.Series([0]*len(team_squad)))))
                            top_players = team_squad.nlargest(5, 'ep')[['web_name', 'position', 'now_cost', 'ep', 'selected_by_percent']]
                            top_players.columns = ['Player', 'Pos', 'Price', 'EP', 'Own%']
                            st.dataframe(top_players, hide_index=True, use_container_width=True)
                else:
                    st.info("No upcoming fixtures found")
    
    except Exception as e:
        st.info(f"Fixture data unavailable: {e}")


def render_price_watch(players_df: pd.DataFrame):
    """Render price watch section."""
    st.markdown('<p class="section-title">Price Watch</p>', unsafe_allow_html=True)
    
    price_df = players_df.copy()
    price_df['transfers_in_event'] = safe_numeric(price_df.get('transfers_in_event', pd.Series([0]*len(price_df))))
    price_df['transfers_out_event'] = safe_numeric(price_df.get('transfers_out_event', pd.Series([0]*len(price_df))))
    price_df['net_transfers'] = price_df['transfers_in_event'] - price_df['transfers_out_event']
    price_df['minutes'] = safe_numeric(price_df.get('minutes', pd.Series([0]*len(price_df))))
    price_df = price_df[price_df['minutes'] > 0]
    
    pr1, pr2 = st.columns(2)
    
    with pr1:
        st.markdown("**Likely to Rise**")
        risers = price_df.nlargest(8, 'net_transfers')[['web_name', 'team_name', 'now_cost', 'net_transfers']]
        risers.columns = ['Player', 'Team', 'Price', 'Net Transfers']
        risers['Net Transfers'] = risers['Net Transfers'].apply(lambda x: f"+{int(x):,}")
        st.dataframe(style_df_with_injuries(risers), hide_index=True, use_container_width=True)
    
    with pr2:
        st.markdown("**Likely to Fall**")
        fallers = price_df.nsmallest(8, 'net_transfers')[['web_name', 'team_name', 'now_cost', 'net_transfers']]
        fallers.columns = ['Player', 'Team', 'Price', 'Net Transfers']
        fallers['Net Transfers'] = fallers['Net Transfers'].apply(lambda x: f"{int(x):,}")
        st.dataframe(style_df_with_injuries(fallers), hide_index=True, use_container_width=True)


def render_injury_alerts(players_df: pd.DataFrame):
    """Render injury alerts section."""
    st.markdown('<p class="section-title">Injury & Doubt Alerts</p>', unsafe_allow_html=True)
    
    df = players_df.copy()
    df['chance_of_playing'] = safe_numeric(df.get('chance_of_playing_next_round', pd.Series([100]*len(df))), 100)
    df['status'] = df.get('status', 'a')
    df['news'] = df.get('news', '').fillna('')
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    flagged = df[
        ((df['status'].isin(['i', 'd', 's', 'u'])) | (df['chance_of_playing'] < 100)) &
        (df['selected_by_percent'] > 1)
    ].sort_values('selected_by_percent', ascending=False)
    
    if not flagged.empty:
        display = flagged.head(15)[['web_name', 'team_name', 'position', 'selected_by_percent', 'chance_of_playing', 'news']].copy()
        display = display.reset_index(drop=True)
        display.columns = ['Player', 'Team', 'Pos', 'Own%', 'Chance', 'News']
        display['Own%'] = display['Own%'].round(1).map(lambda x: f'{x:.1f}')
        # Store numeric chance for styling before converting to string
        chance_values = display['Chance'].astype(int).tolist()
        display['Chance'] = display['Chance'].astype(int).astype(str) + '%'
        
        # Style rows based on injury severity
        def style_injury_rows(df):
            styles = []
            for i in range(len(df)):
                chance = chance_values[i] if i < len(chance_values) else 100
                if chance == 0:
                    bg = 'background-color: rgba(239, 68, 68, 0.4)'  # Red - out
                elif chance <= 25:
                    bg = 'background-color: rgba(239, 68, 68, 0.3)'  # Dark red
                elif chance <= 50:
                    bg = 'background-color: rgba(249, 115, 22, 0.3)'  # Orange
                elif chance <= 75:
                    bg = 'background-color: rgba(245, 158, 11, 0.25)'  # Amber
                else:
                    bg = 'background-color: rgba(250, 204, 21, 0.15)'  # Light yellow
                styles.append([bg] * len(df.columns))
            return pd.DataFrame(styles, index=df.index, columns=df.columns)
        
        styled = display.style.apply(style_injury_rows, axis=None)
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.success("No major injury concerns in popular players")


def render_ownership_trends(players_df: pd.DataFrame):
    """Render ownership trends chart."""
    st.markdown('<p class="section-title">Ownership Trends</p>', unsafe_allow_html=True)
    
    df = players_df.copy()
    df['transfers_in_event'] = safe_numeric(df.get('transfers_in_event', pd.Series([0]*len(df))))
    df['transfers_out_event'] = safe_numeric(df.get('transfers_out_event', pd.Series([0]*len(df))))
    df['total_transfers'] = df['transfers_in_event'] + df['transfers_out_event']
    
    top_movers = df.nlargest(20, 'total_transfers').copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=top_movers['transfers_in_event'],
        name='In',
        marker_color='#22c55e'
    ))
    fig.add_trace(go.Bar(
        x=top_movers['web_name'],
        y=-top_movers['transfers_out_event'],
        name='Out',
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        barmode='relative',
        height=350,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=80),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True, key='strategy_ownership_trends')
