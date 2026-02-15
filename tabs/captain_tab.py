"""Captain Analysis Tab - Dedicated captain decision tool."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric, style_df_with_injuries, normalize_name, search_players
from components.styles import render_section_title, render_divider, render_stat_card


def render_captain_tab(processor, players_df: pd.DataFrame):
    """Captain Analysis tab - comprehensive captain decision tool."""
    
    st.markdown('<p class="section-title">Captain Analysis</p>', unsafe_allow_html=True)
    st.caption("Compare top captain candidates using ML, Poisson, and FPL estimates")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Captain Metrics"):
        st.markdown("""
        **Captain Score**
        - Blended score combining EP (35%), ML prediction (25%), Form (20%), Ownership (10%), and Threat Momentum (10%)
        - Higher score = better captain pick confidence
        
        **Expected Points (EP)**
        - FPL's official prediction for points this GW
        - Based on fixture difficulty, form, and historical data
        
        **ML Prediction**
        - Machine learning model prediction (run ML tab first for accuracy)
        - Uses XGBoost ensemble trained on historical FPL data
        
        **Poisson EP**
        - Statistical model using expected goals (xG) and assists (xA)
        - Better for predicting attacking returns
        
        **Captain EV (Expected Value)**
        - EP × 1.25 (2025/26 captain multiplier)
        - The expected points if you captain this player
        
        **Ownership Consideration**
        - High ownership = safe pick (limits rank loss if they score)
        - Low ownership = differential (big gains if they haul)
        """)
    
    # Prepare data
    df = players_df.copy()
    df['expected_points'] = safe_numeric(df.get('expected_points', df.get('ep_next', pd.Series([2.0]*len(df)))))
    df['ep_next_num'] = safe_numeric(df.get('ep_next_num', df.get('ep_next', pd.Series([2.0]*len(df)))))
    df['form'] = safe_numeric(df.get('form', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0]*len(df))))
    
    # Get ML predictions from session_state if available
    if 'ml_predictions' in st.session_state:
        ml_preds = st.session_state['ml_predictions']
        # ML predictions is a dict of player_id -> prediction object
        df['ml_pred'] = df['id'].apply(
            lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else df.loc[df['id'] == pid, 'expected_points'].iloc[0] if len(df[df['id'] == pid]) > 0 else 2.0
        )
        ml_available = True
    else:
        # Fallback: generate a slightly varied estimate based on form and EP
        df['ml_pred'] = df['expected_points'] * 0.85 + df['form'] * 0.3
        ml_available = False
    
    # Show warning if ML not available
    if not ml_available:
        st.info("Run ML Predictions tab first for accurate ML estimates. Currently showing estimated values.")
    
    # Filter for viable captains (min minutes, reasonable EP)
    viable = df[(df['minutes'] > 200) & (df['expected_points'] > 2)].copy()
    
    # Calculate captain score (blend of multiple signals)
    viable['captain_score'] = (
        viable['expected_points'] * 0.35 +
        viable['ml_pred'] * 0.25 +
        viable['form'] * 0.20 +
        (viable['selected_by_percent'] / 10) * 0.10 +
        safe_numeric(viable.get('threat_momentum', pd.Series([0]*len(viable)))) * 0.10
    )
    
    # Top 10 captain candidates
    top_captains = viable.nlargest(10, 'captain_score')
    
    # ── Top 3 Picks Section ──
    st.markdown("### Top 3 Captain Picks")
    
    pick_cols = st.columns(3)
    pos_colors = {'GKP': '#3b82f6', 'DEF': '#22c55e', 'MID': '#f59e0b', 'FWD': '#ef4444'}
    
    for i, (_, player) in enumerate(top_captains.head(3).iterrows()):
        with pick_cols[i]:
            pos_color = pos_colors.get(player['position'], '#888')
            ep = player['expected_points']
            fpl_ep = player['ep_next_num']
            ml = player['ml_pred']
            form = player['form']
            eo = player['selected_by_percent']
            
            # Differential indicator
            is_diff = eo < 15
            diff_badge = '<span style="color:#22c55e;font-size:0.7rem;margin-left:0.5rem;">DIFFERENTIAL</span>' if is_diff else ''
            
            # Consensus score
            consensus = (ep + fpl_ep + ml) / 3
            
            label = ["Top Pick", "Safe Pick", "Differential"][i] if i < 3 else f"#{i+1}"
            
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
                <div style="color:{pos_color};font-size:0.7rem;font-weight:600;text-transform:uppercase;">{label}</div>
                <div style="color:#1d1d1f;font-size:1.3rem;font-weight:700;margin:0.3rem 0;">{player['web_name']}{diff_badge}</div>
                <div style="color:#86868b;font-size:0.85rem;">{player.get('team_name', '')} | {player['position']} | £{player['now_cost']:.1f}m</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;margin-top:0.75rem;text-align:center;">
                    <div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">Poisson EP</div><div style="color:#3b82f6;font-weight:600;font-family:'JetBrains Mono',monospace;">{ep:.1f}</div></div>
                    <div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">FPL EP</div><div style="color:#22c55e;font-weight:600;font-family:'JetBrains Mono',monospace;">{fpl_ep:.1f}</div></div>
                    <div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">ML Pred</div><div style="color:#f59e0b;font-weight:600;font-family:'JetBrains Mono',monospace;">{ml:.1f}</div></div>
                    <div><div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">Form</div><div style="color:#1d1d1f;font-weight:600;font-family:'JetBrains Mono',monospace;">{form:.1f}</div></div>
                </div>
                <div style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid rgba(0,0,0,0.08);">
                    <div style="color:#86868b;font-size:0.65rem;text-transform:uppercase;">Consensus</div>
                    <div style="color:#1d1d1f;font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{consensus:.1f} pts</div>
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
        display_df = filtered[['web_name', 'team_name', 'position', 'now_cost', 'expected_points', 
                               'ep_next_num', 'ml_pred', 'form', 'selected_by_percent', 'captain_score']].copy()
        display_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Poisson EP', 'FPL EP', 'ML Pred', 'Form', 'EO%', 'Score']
        
        st.dataframe(
            style_df_with_injuries(display_df, players_df, format_dict={
                'Price': '£{:.1f}m',
                'Poisson EP': '{:.2f}',
                'FPL EP': '{:.2f}',
                'ML Pred': '{:.2f}',
                'Form': '{:.1f}',
                'EO%': '{:.1f}%',
                'Score': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # ── Head-to-Head Comparison ──
    st.markdown("### Head-to-Head Comparison")
    
    # Create normalized name column for searching
    viable['name_normalized'] = viable['web_name'].apply(lambda x: normalize_name(str(x).lower()))
    
    h2h_cols = st.columns(2)
    
    with h2h_cols[0]:
        search1 = st.text_input("Search Player 1", key="h2h_search1", placeholder="Type player name...")
        # Show suggestions dropdown
        if search1 and len(search1) >= 2:
            query1_norm = normalize_name(search1.lower())
            suggestions1 = viable[viable['name_normalized'].str.contains(query1_norm, na=False)].head(5)
            if not suggestions1.empty:
                player1_select = st.selectbox(
                    "Select player:",
                    options=suggestions1['web_name'].tolist(),
                    key="h2h_select1",
                    label_visibility="collapsed"
                )
            else:
                player1_select = None
                st.caption("No matches found")
        else:
            player1_select = None
            
    with h2h_cols[1]:
        search2 = st.text_input("Search Player 2", key="h2h_search2", placeholder="Type player name...")
        # Show suggestions dropdown
        if search2 and len(search2) >= 2:
            query2_norm = normalize_name(search2.lower())
            suggestions2 = viable[viable['name_normalized'].str.contains(query2_norm, na=False)].head(5)
            if not suggestions2.empty:
                player2_select = st.selectbox(
                    "Select player:",
                    options=suggestions2['web_name'].tolist(),
                    key="h2h_select2",
                    label_visibility="collapsed"
                )
            else:
                player2_select = None
                st.caption("No matches found")
        else:
            player2_select = None
    
    # Use selected players
    player1 = player1_select
    player2 = player2_select
    
    if player1 and player2 and player1 != player2:
        p1_data = viable[viable['web_name'] == player1].iloc[0] if not viable[viable['web_name'] == player1].empty else None
        p2_data = viable[viable['web_name'] == player2].iloc[0] if not viable[viable['web_name'] == player2].empty else None
        
        if p1_data is not None and p2_data is not None:
            # Radar chart comparison
            categories = ['Poisson EP', 'FPL EP', 'ML Pred', 'Form', 'Ceiling']
            
            # Normalize values for radar
            max_ep = max(p1_data['expected_points'], p2_data['expected_points'], 1)
            max_fpl = max(p1_data['ep_next_num'], p2_data['ep_next_num'], 1)
            max_ml = max(p1_data['ml_pred'], p2_data['ml_pred'], 1)
            max_form = max(p1_data['form'], p2_data['form'], 1)
            
            # Ceiling estimate (EP + form bonus)
            p1_ceiling = p1_data['expected_points'] * 1.5 + p1_data['form']
            p2_ceiling = p2_data['expected_points'] * 1.5 + p2_data['form']
            max_ceiling = max(p1_ceiling, p2_ceiling, 1)
            
            p1_values = [
                p1_data['expected_points'] / max_ep * 100,
                p1_data['ep_next_num'] / max_fpl * 100,
                p1_data['ml_pred'] / max_ml * 100,
                p1_data['form'] / max_form * 100,
                p1_ceiling / max_ceiling * 100
            ]
            p2_values = [
                p2_data['expected_points'] / max_ep * 100,
                p2_data['ep_next_num'] / max_fpl * 100,
                p2_data['ml_pred'] / max_ml * 100,
                p2_data['form'] / max_form * 100,
                p2_ceiling / max_ceiling * 100
            ]
            
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
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#333'),
                    bgcolor='#ffffff'
                ),
                showlegend=True,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                font=dict(family='Inter, sans-serif', color='#fff'),
                height=350,
                margin=dict(l=60, r=60, t=30, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Side by side metrics
            m1, m2 = st.columns(2)
            
            with m1:
                winner_ep = "✓" if p1_data['expected_points'] >= p2_data['expected_points'] else ""
                winner_fpl = "✓" if p1_data['ep_next_num'] >= p2_data['ep_next_num'] else ""
                winner_ml = "✓" if p1_data['ml_pred'] >= p2_data['ml_pred'] else ""
                winner_form = "✓" if p1_data['form'] >= p2_data['form'] else ""
                
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;">
                    <div style="color:#3b82f6;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">{player1}</div>
                    <div style="display:grid;gap:0.3rem;">
                        <div style="display:flex;justify-content:space-between;"><span>Poisson EP:</span><span style="color:#1d1d1f;">{p1_data['expected_points']:.1f} {winner_ep}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>FPL EP:</span><span style="color:#1d1d1f;">{p1_data['ep_next_num']:.1f} {winner_fpl}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>ML Pred:</span><span style="color:#1d1d1f;">{p1_data['ml_pred']:.1f} {winner_ml}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>Form:</span><span style="color:#1d1d1f;">{p1_data['form']:.1f} {winner_form}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>EO:</span><span style="color:#1d1d1f;">{p1_data['selected_by_percent']:.1f}%</span></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with m2:
                winner_ep = "✓" if p2_data['expected_points'] >= p1_data['expected_points'] else ""
                winner_fpl = "✓" if p2_data['ep_next_num'] >= p1_data['ep_next_num'] else ""
                winner_ml = "✓" if p2_data['ml_pred'] >= p1_data['ml_pred'] else ""
                winner_form = "✓" if p2_data['form'] >= p1_data['form'] else ""
                
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;">
                    <div style="color:#ef4444;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">{player2}</div>
                    <div style="display:grid;gap:0.3rem;">
                        <div style="display:flex;justify-content:space-between;"><span>Poisson EP:</span><span style="color:#1d1d1f;">{p2_data['expected_points']:.1f} {winner_ep}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>FPL EP:</span><span style="color:#1d1d1f;">{p2_data['ep_next_num']:.1f} {winner_fpl}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>ML Pred:</span><span style="color:#1d1d1f;">{p2_data['ml_pred']:.1f} {winner_ml}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>Form:</span><span style="color:#1d1d1f;">{p2_data['form']:.1f} {winner_form}</span></div>
                        <div style="display:flex;justify-content:space-between;"><span>EO:</span><span style="color:#1d1d1f;">{p2_data['selected_by_percent']:.1f}%</span></div>
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
    
    diff_captains = viable[(viable['selected_by_percent'] < 10) & (viable['expected_points'] > 3)].nlargest(8, 'captain_score')
    
    if not diff_captains.empty:
        diff_cols = st.columns(4)
        for i, (_, p) in enumerate(diff_captains.head(4).iterrows()):
            with diff_cols[i]:
                pos_color = pos_colors.get(p['position'], '#888')
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:0.75rem;text-align:center;">
                    <div style="color:{pos_color};font-size:0.7rem;">{p['position']}</div>
                    <div style="color:#1d1d1f;font-weight:600;">{p['web_name']}</div>
                    <div style="color:#86868b;font-size:0.8rem;">{p.get('team_name', '')} | £{p['now_cost']:.1f}m</div>
                    <div style="color:#22c55e;font-size:0.9rem;margin-top:0.3rem;">EP: {p['expected_points']:.2f}</div>
                    <div style="color:#f59e0b;font-size:0.8rem;">EO: {p['selected_by_percent']:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No viable differential captains found")
    
    # ── Historical Captain Success Rate ──
    st.markdown("---")
    st.markdown("### Captain Success Metrics")
    st.caption("Based on form momentum and recent returns")
    
    # Form trend visualization
    form_df = viable.nlargest(8, 'captain_score').copy()
    
    if not form_df.empty and 'form' in form_df.columns:
        fig = go.Figure()
        
        for i, (_, p) in enumerate(form_df.iterrows()):
            pos_color = pos_colors.get(p['position'], '#888')
            form_val = p['form']
            ep_val = p['expected_points']
            
            fig.add_trace(go.Bar(
                x=[p['web_name']],
                y=[form_val],
                name='Form',
                marker_color=pos_color,
                opacity=0.8,
                showlegend=(i == 0),
                legendgroup='form',
                hovertemplate=f"<b>{p['web_name']}</b><br>Form: {form_val:.1f}<br>EP: {ep_val:.2f}<extra></extra>"
            ))
        
        fig.update_layout(
            height=300,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(gridcolor='#e5e5ea', tickangle=45),
            yaxis=dict(title='Form', gridcolor='#e5e5ea'),
            margin=dict(l=50, r=30, t=20, b=80),
            showlegend=False,
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
