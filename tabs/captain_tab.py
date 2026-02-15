"""Captain Analysis Tab - Dedicated captain decision tool."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import (
    safe_numeric, style_df_with_injuries, normalize_name, 
    search_players, calculate_consensus_ep, get_consensus_label
)
from components.styles import render_section_title, render_divider, render_stat_card


def render_captain_tab(processor, players_df: pd.DataFrame, fetcher):
    """Captain Analysis tab - comprehensive captain decision tool."""
    
    st.markdown('<p class="section-title">Captain Analysis</p>', unsafe_allow_html=True)
    st.caption("Compare top captain candidates using ML, Poisson, and FPL xP estimates")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Captain Metrics"):
        st.markdown("""
        **Captain Score**
        - Blended score combining xP (35%), ML xP (25%), Form (20%), Ownership (10%), and Threat Momentum (10%)
        - Higher score = better captain pick confidence
        
        **Model xP (Standardized)**
        - Consensus prediction for points this GW
        - Based on fixture difficulty, form, and historical data
        
        **ML xP**
        - Machine learning model prediction (run ML tab first for accuracy)
        - Uses XGBoost ensemble trained on historical FPL data
        
        **Poisson xP**
        - Statistical model using expected goals (xG) and assists (xA)
        - Better for predicting attacking returns
        
        **Captain xP (Expected Value)**
        - Model xP × 1.25 (2025/26 captain multiplier)
        - The xP if you captain this player
        
        **Global Model xP Weights**
        - **Model xP**: Weighted average of all enabled models.
        - **Captain Score**: 60% Core Models, 30% Position Potential, 10% Meta.
        - *Core (3 enabled)*: 35% Poisson xP, 25% ML xP.
        - *Core (2 enabled)*: Re-weights 60% among selections.
        
        **Ownership Consideration**
        - High ownership = safe pick (limits rank loss if they score)
        - Low ownership = differential (big gains if they haul)
        """)
        
    # Use Global Model Selection
    active_models_keys = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    
    enable_poisson = 'poisson' in active_models_keys
    enable_fpl = 'fpl' in active_models_keys
    enable_ml = 'ml' in active_models_keys
    
    active_models = []
    if enable_poisson: active_models.append('poisson_ep')
    if enable_fpl: active_models.append('ep_next_num')
    if enable_ml: active_models.append('ml_pred')
    
    # Prepare data
    df = players_df.copy()
    
    # Poisson EP specifically for the captain model comparison
    if 'expected_points_poisson' in df.columns:
        df['poisson_ep'] = safe_numeric(df['expected_points_poisson'])
    else:
        df['poisson_ep'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([2.0]*len(df)))))
        
    df['expected_points'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([2.0]*len(df)))))
    df['ep_next_num'] = safe_numeric(df.get('ep_next_num', df.get('ep_next', pd.Series([2.0]*len(df)))))
    df['form'] = safe_numeric(df.get('form', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    
    # Get ML predictions from session_state if available
    if 'ml_predictions' in st.session_state:
        ml_preds = st.session_state['ml_predictions']
        df['ml_pred'] = df['id'].apply(
            lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else df.loc[df['id'] == pid, 'expected_points'].iloc[0] if len(df[df['id'] == pid]) > 0 else 2.0
        )
        ml_available = True
    else:
        df['ml_pred'] = df['expected_points'] * 0.85 + df['form'] * 0.3
        ml_available = False
    
    # Calculate Global Model xP (Consensus) first
    active_m = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    df = calculate_consensus_ep(df, active_m)
    
    # Consider all players in the database as viable captain candidates
    viable = df.copy()
    
    # Calculate enhanced position-aware captain score
    from utils.helpers import calculate_enhanced_captain_score
    viable['captain_score'] = viable.apply(
        lambda r: calculate_enhanced_captain_score(r, active_models), axis=1
    ).round(2)

    # Use pre-calculated consensus_ep (Model xP)
    viable['consensus'] = safe_numeric(viable['consensus_ep'])
    viable['captain_ev'] = viable['consensus'] * 1.25 # Captain Multiplier
    
    # ── Your Team's Best Picks ──
    team_id = st.session_state.get('fpl_team_id', 0)
    if team_id > 0:
        st.markdown("### Your Team's Best Picks")
        try:
            gw = fetcher.get_current_gameweek()
            picks_data = fetcher.get_team_picks(team_id, gw)
            if isinstance(picks_data, dict) and 'picks' in picks_data:
                your_squad_ids = [p['element'] for p in picks_data['picks']]
            elif isinstance(picks_data, list):
                your_squad_ids = [p.get('element', p) for p in picks_data]
            else:
                your_squad_ids = []
            
            if your_squad_ids:
                your_viable = viable[viable['id'].isin(your_squad_ids)].nlargest(3, 'captain_score')
                if not your_viable.empty:
                    ycols = st.columns(3)
                    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
                    for i, (_, player) in enumerate(your_viable.iterrows()):
                        with ycols[i]:
                            pos_color = pos_colors.get(player['position'], '#888')
                            eo = player['selected_by_percent']
                            is_diff = eo < 15
                            diff_badge = '<span style="color:#22c55e;font-size:0.7rem;margin-left:0.5rem;">DIFFERENTIAL</span>' if is_diff else ''
                            
                            st.markdown(f'''
                            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                                <div style="color:#1d1d1f;font-size:0.7rem;font-weight:600;text-transform:uppercase;">Squad Recommendation #{i+1}</div>
                                <div style="color:#1d1d1f;font-size:1.3rem;font-weight:700;margin:0.3rem 0;">{player['web_name']}{diff_badge}</div>
                                <div style="color:#86868b;font-size:0.85rem;">{player.get('team_name', '')} | {player['position']}</div>
                                <div style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid rgba(0,0,0,0.08);">
                                    <div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">{get_consensus_label(st.session_state.active_models)} Prediction</div>
                                    <div style="color:#1d1d1f;font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{player['consensus']:.1f}</div>
                                </div>
                                <div style="color:#86868b;font-size:0.7rem;margin-top:0.3rem;">Captain Score: {player['captain_score']:.2f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                else:
                    st.info("No viable captain candidates found in your squad.")
            else:
                st.warning(f"Could not find squad data for Team ID {team_id}. Make sure it's public.")
        except Exception as e:
            st.error(f"Error fetching your squad: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
    top_captains = viable.nlargest(10, 'captain_score')
    
    # ── Top 3 Picks Section ──
    st.markdown("### Top 3 Captain Picks")
    
    pick_cols = st.columns(3)
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    for i, (_, player) in enumerate(top_captains.head(3).iterrows()):
        with pick_cols[i]:
            pos_color = pos_colors.get(player['position'], '#888')
            ep_poisson = player['poisson_ep']
            fpl_ep = player['ep_next_num']
            ml = player['ml_pred']
            form = player['form']
            eo = player['selected_by_percent']
            
            # Differential indicator
            is_diff = eo < 15
            diff_badge = '<span style="color:#22c55e;font-size:0.7rem;margin-left:0.5rem;">DIFFERENTIAL</span>' if is_diff else ''
            
            # Consensus score
            consensus = player['consensus']
            
            label = ["Top Pick", "Safe Pick", "Differential"][i] if i < 3 else f"#{i+1}"
            
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem 1rem 0 1rem;text-align:center;border-bottom:none;border-bottom-left-radius:0;border-bottom-right-radius:0;">
                <div style="color:#1d1d1f;font-size:0.7rem;font-weight:600;text-transform:uppercase;">{label}</div>
                <div style="color:#1d1d1f;font-size:1.3rem;font-weight:700;margin:0.3rem 0;">{player['web_name']}{diff_badge}</div>
                <div style="color:#86868b;font-size:0.85rem;">{player.get('team_name', '')} | {player['position']} | £{player['now_cost']:.1f}m</div>
            </div>
            ''', unsafe_allow_html=True)

            # Prepare Stat Grid items based on enabled models
            stat_grid_items = []
            if enable_poisson:
                stat_grid_items.append(f'<div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">Poisson xP</div><div style="color:#3b82f6;font-weight:600;font-family:\'JetBrains Mono\',monospace;">{ep_poisson:.1f}</div></div>')
            if enable_fpl:
                stat_grid_items.append(f'<div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">FPL xP</div><div style="color:#22c55e;font-weight:600;font-family:\'JetBrains Mono\',monospace;">{fpl_ep:.1f}</div></div>')
            if enable_ml:
                stat_grid_items.append(f'<div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">ML xP</div><div style="color:#f59e0b;font-weight:600;font-family:\'JetBrains Mono\',monospace;">{ml:.1f}</div></div>')
            stat_grid_items.append(f'<div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">Form</div><div style="color:#1d1d1f;font-weight:600;font-family:\'JetBrains Mono\',monospace;">{form:.1f}</div></div>')
            
            grid_columns = 2
            stat_grid_html = f'<div style="display:grid;grid-template-columns:repeat({grid_columns},1fr);gap:0.5rem;padding:0 1rem;text-align:center;background:#fff;border-left:1px solid rgba(0,0,0,0.04);border-right:1px solid rgba(0,0,0,0.04);">' + "".join(stat_grid_items) + '</div>'
            
            st.markdown(f'''
                {stat_grid_html}
                <div style="background:#fff;border:1px solid rgba(0,0,0,0.04);border-top:none;border-radius:10px;border-top-left-radius:0;border-top-right-radius:0;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:0 1rem 1rem 1rem;text-align:center;">
                    <div style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid rgba(0,0,0,0.08);">
                        <div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">{get_consensus_label(st.session_state.active_models)}</div>
                        <div style="color:#1d1d1f;font-size:1.1rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{consensus:.1f}</div>
                    </div>
                    <div style="color:#86868b;font-size:0.7rem;margin-top:0.3rem;">EO: {eo:.1f}%</div>
                </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Comparison Table ──
    st.markdown("### Full Captain Comparison")
    
    # Filters
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        cap_pos = st.selectbox("Position", ['All', 'GKP', 'DEF', 'MID', 'FWD'], key="cap_pos")
    with fc2:
        cap_type = st.selectbox("Type", ['All', 'Premium (>9m)', 'Mid-range (6-9m)', 'Budget (<6m)'], key="cap_type")
    with fc3:
        cap_search = st.text_input("Search player", key="cap_search")
    
    filtered = top_captains.copy() if len(top_captains) > 0 else viable.nlargest(10, 'expected_points')
    
    # Add normalized names for searching
    viable['name_normalized'] = viable['web_name'].apply(lambda x: normalize_name(str(x).lower()))
    
    # Apply filters
    if cap_pos != 'All':
        filtered = viable[viable['position'] == cap_pos].nlargest(10, 'captain_score')
    
    if cap_type == 'Premium (>9m)':
        filtered = filtered[filtered['now_cost'] > 9]
    elif cap_type == 'Mid-range (6-9m)':
        filtered = filtered[(filtered['now_cost'] >= 6) & (filtered['now_cost'] <= 9)]
    elif cap_type == 'Budget (<6m)':
        filtered = filtered[filtered['now_cost'] < 6]
    
    if cap_search:
        query_norm = normalize_name(cap_search.lower())
        filtered = viable[viable['name_normalized'].str.contains(query_norm, na=False)].nlargest(10, 'captain_score')
    
    if not filtered.empty:
        # Build dynamic columns for display_df
        cols_to_keep = ['web_name', 'team_name', 'position', 'now_cost']
        col_names = ['Player', 'Team', 'Pos', 'Price']
        format_dict = {'Price': '£{:.1f}m', 'Form': '{:.1f}', 'EO%': '{:.1f}%', 'Score': '{:.2f}', 'Consensus': '{:.2f}'}
        
        if enable_poisson:
            cols_to_keep.append('poisson_ep')
            col_names.append('Poisson EP')
            format_dict['Poisson EP'] = '{:.2f}'
        if enable_fpl:
            cols_to_keep.append('ep_next_num')
            col_names.append('FPL EP')
            format_dict['FPL EP'] = '{:.2f}'
        if enable_ml:
            cols_to_keep.append('ml_pred')
            col_names.append('ML Pred')
            format_dict['ML Pred'] = '{:.2f}'
            
        cols_to_keep.extend(['consensus', 'form', 'selected_by_percent', 'captain_score'])
        col_names.extend(['Consensus', 'Form', 'EO%', 'Score'])
        
        display_df = filtered[cols_to_keep].copy()
        display_df.columns = col_names
        
        st.dataframe(
            style_df_with_injuries(display_df, players_df, format_dict=format_dict),
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # ── Head-to-Head Comparison ──
    st.markdown("### Head-to-Head Comparison")
    
    # Create normalized components for searching
    if 'first_normalized' not in viable.columns:
        viable['first_normalized'] = viable['first_name'].apply(lambda x: normalize_name(str(x).lower()))
        viable['second_normalized'] = viable['second_name'].apply(lambda x: normalize_name(str(x).lower()))
    
    h2h_cols = st.columns(2)
    
    with h2h_cols[0]:
        search1 = st.text_input("Search Player 1", key="h2h_search1", placeholder="Search first or last name...")
        # Show suggestions dropdown
        if search1 and len(search1) >= 1:
            q1 = normalize_name(search1.lower().strip())
            suggestions1 = viable[
                (viable['first_normalized'].str.startswith(q1, na=False)) |
                (viable['second_normalized'].str.startswith(q1, na=False))
            ].sort_values('selected_by_percent', ascending=False).head(10)
            
            if not suggestions1.empty:
                player1_select = st.selectbox(
                    "Select player:",
                    options=suggestions1['full_name'].tolist(),
                    key="h2h_select1",
                    label_visibility="collapsed"
                )
            else:
                player1_select = None
                st.caption("No matches found (starts with)")
        else:
            player1_select = None
            
    with h2h_cols[1]:
        search2 = st.text_input("Search Player 2", key="h2h_search2", placeholder="Search first or last name...")
        # Show suggestions dropdown
        if search2 and len(search2) >= 1:
            q2 = normalize_name(search2.lower().strip())
            suggestions2 = viable[
                (viable['first_normalized'].str.startswith(q2, na=False)) |
                (viable['second_normalized'].str.startswith(q2, na=False))
            ].sort_values('selected_by_percent', ascending=False).head(10)
            
            if not suggestions2.empty:
                player2_select = st.selectbox(
                    "Select player:",
                    options=suggestions2['full_name'].tolist(),
                    key="h2h_select2",
                    label_visibility="collapsed"
                )
            else:
                player2_select = None
                st.caption("No matches found (starts with)")
        else:
            player2_select = None
    
    # Use selected players
    player1 = player1_select
    player2 = player2_select
    
    if player1 and player2 and player1 != player2:
        p1_data = viable[viable['full_name'] == player1].iloc[0] if not viable[viable['full_name'] == player1].empty else None
        p2_data = viable[viable['full_name'] == player2].iloc[0] if not viable[viable['full_name'] == player2].empty else None
        
        if p1_data is not None and p2_data is not None:
            # Radar chart comparison
            categories = []
            if enable_poisson: categories.append('Poisson xP')
            if enable_fpl: categories.append('FPL xP')
            if enable_ml: categories.append('ML xP')
            categories.extend(['Recent Form', 'Matchup Ease', 'Bonus Potential'])
            
            # Helper to get bonus potential based on position
            def get_bonus(p):
                if p['position'] in ['GKP', 'DEF']:
                    return safe_numeric(pd.Series([p.get('cbit_prob', 0.5)])).iloc[0] * 100
                return safe_numeric(pd.Series([p.get('threat_momentum', 0.5)])).iloc[0] * 100

            # Max values for normalization
            max_vals = {
                'poisson_ep': max(p1_data['poisson_ep'], p2_data['poisson_ep'], 1),
                'ep_next_num': max(p1_data['ep_next_num'], p2_data['ep_next_num'], 1),
                'ml_pred': max(p1_data['ml_pred'], p2_data['ml_pred'], 1),
                'form': max(p1_data['form'], p2_data['form'], 1),
                'matchup': max(p1_data.get('matchup_quality', 1), p2_data.get('matchup_quality', 1), 1)
            }
            
            p1_bonus = get_bonus(p1_data)
            p2_bonus = get_bonus(p2_data)
            max_bonus = max(p1_bonus, p2_bonus, 1)

            p1_values = []
            p2_values = []
            if enable_poisson:
                p1_values.append((p1_data['poisson_ep'] / max_vals['poisson_ep']) * 100)
                p2_values.append((p2_data['poisson_ep'] / max_vals['poisson_ep']) * 100)
            if enable_fpl:
                p1_values.append((p1_data['ep_next_num'] / max_vals['ep_next_num']) * 100)
                p2_values.append((p2_data['ep_next_num'] / max_vals['ep_next_num']) * 100)
            if enable_ml:
                p1_values.append((p1_data['ml_pred'] / max_vals['ml_pred']) * 100)
                p2_values.append((p2_data['ml_pred'] / max_vals['ml_pred']) * 100)
                
            p1_values.extend([
                (p1_data['form'] / max_vals['form']) * 100,
                (p1_data.get('matchup_quality', 1) / max_vals['matchup']) * 100,
                (p1_bonus / max_bonus) * 100
            ])
            p2_values.extend([
                (p2_data['form'] / max_vals['form']) * 100,
                (p2_data.get('matchup_quality', 1) / max_vals['matchup']) * 100,
                (p2_bonus / max_bonus) * 100
            ])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=p1_values + [p1_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=player1,
                line_color='#3b82f6',
                fillcolor='rgba(59,130,246,0.3)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=p2_values + [p2_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=player2,
                line_color='#ef4444',
                fillcolor='rgba(239,68,68,0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#e5e5ea'),
                    angularaxis=dict(gridcolor='#e5e5ea', linecolor='#e5e5ea'),
                    bgcolor='#ffffff'
                ),
                showlegend=True,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                font=dict(family='Inter, sans-serif', color='#444', size=11),
                height=350,
                margin=dict(l=80, r=80, t=30, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Side by side metrics
            m1, m2 = st.columns(2)
            
            with m1:
                items = []
                if enable_poisson:
                    win = "✓" if p1_data['poisson_ep'] >= p2_data['poisson_ep'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>Poisson EP:</span><span style="color:#1d1d1f;">{p1_data["poisson_ep"]:.1f} {win}</span></div>')
                if enable_fpl:
                    win = "✓" if p1_data['ep_next_num'] >= p2_data['ep_next_num'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>FPL EP:</span><span style="color:#1d1d1f;">{p1_data["ep_next_num"]:.1f} {win}</span></div>')
                if enable_ml:
                    win = "✓" if p1_data['ml_pred'] >= p2_data['ml_pred'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>ML Pred:</span><span style="color:#1d1d1f;">{p1_data["ml_pred"]:.1f} {win}</span></div>')
                
                win_form = "✓" if p1_data['form'] >= p2_data['form'] else ""
                items.append(f'<div style="display:flex;justify-content:space-between;"><span>Form:</span><span style="color:#1d1d1f;">{p1_data["form"]:.1f} {win_form}</span></div>')
                items.append(f'<div style="display:flex;justify-content:space-between;"><span>EO:</span><span style="color:#1d1d1f;">{p1_data["selected_by_percent"]:.1f}%</span></div>')
                
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;">
                    <div style="color:#3b82f6;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">{player1}</div>
                    <div style="display:grid;gap:0.3rem;">
                        {"".join(items)}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with m2:
                items = []
                if enable_poisson:
                    win = "✓" if p2_data['poisson_ep'] >= p1_data['poisson_ep'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>Poisson EP:</span><span style="color:#1d1d1f;">{p2_data["poisson_ep"]:.1f} {win}</span></div>')
                if enable_fpl:
                    win = "✓" if p2_data['ep_next_num'] >= p1_data['ep_next_num'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>FPL EP:</span><span style="color:#1d1d1f;">{p2_data["ep_next_num"]:.1f} {win}</span></div>')
                if enable_ml:
                    win = "✓" if p2_data['ml_pred'] >= p1_data['ml_pred'] else ""
                    items.append(f'<div style="display:flex;justify-content:space-between;"><span>ML Pred:</span><span style="color:#1d1d1f;">{p2_data["ml_pred"]:.1f} {win}</span></div>')
                
                win_form = "✓" if p2_data['form'] >= p1_data['form'] else ""
                items.append(f'<div style="display:flex;justify-content:space-between;"><span>Form:</span><span style="color:#1d1d1f;">{p2_data["form"]:.1f} {win_form}</span></div>')
                items.append(f'<div style="display:flex;justify-content:space-between;"><span>EO:</span><span style="color:#1d1d1f;">{p2_data["selected_by_percent"]:.1f}%</span></div>')
                
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;">
                    <div style="color:#ef4444;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">{player2}</div>
                    <div style="display:grid;gap:0.3rem;">
                        {"".join(items)}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    elif not search1 or not search2:
        st.info("Enter two player names above to compare them head-to-head")
    elif player1 == player2:
        st.warning("Please search for two different players")
    
    st.markdown("---")
    
    # ── Differential Captain Picks ──
    st.markdown("### Differential Captain Picks")
    st.caption("Low ownership (<10%) with high ceiling potential")
    
    diff_captains = viable[(viable['selected_by_percent'] < 10) & (safe_numeric(viable['consensus_ep']) > 3)].nlargest(8, 'captain_score')
    
    if not diff_captains.empty:
        diff_cols = st.columns(4)
        for i, (_, p) in enumerate(diff_captains.head(4).iterrows()):
            with diff_cols[i]:
                pos_color = pos_colors.get(p['position'], '#888')
                ep_val = p['consensus']
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:0.75rem;text-align:center;">
                    <div style="color:{pos_color};font-size:0.7rem;">{p['position']}</div>
                    <div style="color:#1d1d1f;font-weight:600;">{p['web_name']}</div>
                    <div style="color:#86868b;font-size:0.8rem;">{p.get('team_name', '')} | £{p['now_cost']:.1f}m</div>
                    <div style="color:#22c55e;font-size:0.9rem;margin-top:0.3rem;">Consensus: {ep_val:.2f}</div>
                    <div style="color:#f59e0b;font-size:0.8rem;">EO: {p['selected_by_percent']:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No viable differential captains found")
    
    # ── Historical Captain Success Rate ──
    st.markdown("---")
    # ── Comparison of Captain Types ──
    st.markdown("---")
    st.markdown("### Top Captains by Category")
    st.caption("Comparison of Safe (EO>35%), Differential (EO<15%), and overall Top candidates")
    
    # Identify categories
    safe_caps = viable[viable['selected_by_percent'] >= 35].nlargest(5, 'captain_score').copy()
    diff_caps = viable[viable['selected_by_percent'] < 15].nlargest(5, 'captain_score').copy()
    top_caps = viable.nlargest(5, 'captain_score').copy()
    
    # Combine for visualization
    plot_data = []
    
    for _, p in top_caps.iterrows():
        plot_data.append({'Player': p['web_name'], 'Score': p['captain_score'], 'Category': 'Top Pick', 'Color': '#3b82f6'})
    
    for _, p in safe_caps.iterrows():
        plot_data.append({'Player': p['web_name'], 'Score': p['captain_score'], 'Category': 'Safe (Template)', 'Color': '#22c55e'})
        
    for _, p in diff_caps.iterrows():
        plot_data.append({'Player': p['web_name'], 'Score': p['captain_score'], 'Category': 'Differential', 'Color': '#f59e0b'})
        
    if plot_data:
        pdf = pd.DataFrame(plot_data)
        
        fig = go.Figure()
        
        categories = pdf['Category'].unique()
        for cat in categories:
            cat_df = pdf[pdf['Category'] == cat]
            fig.add_trace(go.Bar(
                x=cat_df['Player'],
                y=cat_df['Score'],
                name=cat,
                marker_color=cat_df['Color'].iloc[0],
                hovertemplate="<b>%{x}</b><br>Category: " + cat + "<br>Score: %{y:.2f}<extra></extra>"
            ))
            
        fig.update_layout(
            height=400,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(title='Player', gridcolor='#e5e5ea', tickangle=45),
            yaxis=dict(title='Captaincy Score', gridcolor='#e5e5ea'),
            margin=dict(l=50, r=30, t=20, b=100),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate comparison chart")
