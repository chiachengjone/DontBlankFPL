"""Historical Performance Tab - Track actual weekly scores and decisions."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import safe_numeric
from components.styles import render_section_title


def render_history_tab(processor, players_df: pd.DataFrame, fetcher):
    """Historical Performance tab - track scores and validate ML predictions."""
    
    st.markdown('<p class="section-title">Historical Performance</p>', unsafe_allow_html=True)
    st.caption("Track your GW scores, captain choices, and ML prediction accuracy")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Historical Metrics"):
        st.markdown("""
        **Season Summary**
        - **Total Points**: Your cumulative score this season
        - **Overall Rank**: Your position among all ~11M managers
        - **GW Average**: Average points per gameweek
        - **Best/Worst GW**: Your highest and lowest scoring weeks
        
        **Transfer History**
        - **Total Transfers**: Number of transfers made all season
        - **Hits Taken**: Points spent on extra transfers (-4 each)
        - **Net Transfer Cost**: Total hit points lost
        
        **Captain Analysis**
        - Shows captain choices and returns per GW
        - Captain Points: Points earned from captain pick
        - Optimal Captain: Best captain choice in hindsight
        - Captain Efficiency: Actual vs optimal captain points %
        
        **ML Prediction Accuracy**
        - Compares ML predictions to actual outcomes
        - MAE (Mean Absolute Error): Average prediction miss
        - Hit Rate: % of predictions within 2 points
        
        **Chip Usage**
        - Shows when chips were played and their impact
        - Useful for planning future chip strategy
        """)
    
    # Get team ID
    team_id = st.session_state.get('fpl_team_id', 0)
    
    if team_id == 0:
        st.warning("Enter your FPL Team ID in the header to view historical performance")
        
        # Show general performance analysis instead
        render_general_analysis(processor, players_df, fetcher)
        return
    
    # Try to fetch historical data
    try:
        with st.spinner("Loading your history..."):
            history = fetcher.get_team_history(team_id)
            current_gw = fetcher.get_current_gameweek()
    except Exception as e:
        st.error(f"Could not load history: {e}")
        render_general_analysis(processor, players_df, fetcher)
        return
    
    if not history or 'current' not in history:
        st.warning("No historical data found for this Team ID")
        render_general_analysis(processor, players_df, fetcher)
        return
    
    gw_history = history.get('current', [])
    
    if not gw_history:
        st.info("No gameweek history available yet")
        return
    
    # ── Season Summary ──
    st.markdown("### Season Summary")
    
    total_points = sum(gw.get('points', 0) for gw in gw_history)
    total_transfers = sum(gw.get('event_transfers', 0) for gw in gw_history)
    total_hits = sum(gw.get('event_transfers_cost', 0) for gw in gw_history)
    avg_gw_points = total_points / len(gw_history) if gw_history else 0
    current_rank = gw_history[-1].get('overall_rank', 0) if gw_history else 0
    best_gw = max(gw_history, key=lambda x: x.get('points', 0)) if gw_history else {}
    worst_gw = min(gw_history, key=lambda x: x.get('points', 0)) if gw_history else {}
    
    sum_cols = st.columns(4)
    
    with sum_cols[0]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Total Points</div>
            <div style="color:#3b82f6;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{total_points}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[1]:
        rank_color = '#22c55e' if current_rank < 100000 else '#f59e0b' if current_rank < 500000 else '#6b6b6b'
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Overall Rank</div>
            <div style="color:{rank_color};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{current_rank:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[2]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Avg GW Points</div>
            <div style="color:#22c55e;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{avg_gw_points:.1f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[3]:
        st.markdown(f'''
        <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;text-align:center;">
            <div style="color:#86868b;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Transfer Hits</div>
            <div style="color:#ef4444;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">-{total_hits}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── GW-by-GW Performance Chart ──
    st.markdown("### Gameweek Performance")
    
    gws = [gw.get('event', i+1) for i, gw in enumerate(gw_history)]
    points = [gw.get('points', 0) for gw in gw_history]
    ranks = [gw.get('overall_rank', 0) for gw in gw_history]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Points bars
    fig.add_trace(
        go.Bar(x=gws, y=points, name='Points', marker_color='#3b82f6', opacity=0.8),
        secondary_y=False
    )
    
    # Average line
    avg_line = [avg_gw_points] * len(gws)
    fig.add_trace(
        go.Scatter(x=gws, y=avg_line, name='Average', line=dict(color='#22c55e', dash='dash'), mode='lines'),
        secondary_y=False
    )
    
    # Rank line
    fig.add_trace(
        go.Scatter(x=gws, y=ranks, name='Rank', line=dict(color='#f59e0b'), mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=50, t=40, b=50)
    )
    
    fig.update_xaxes(title_text='Gameweek', gridcolor='#e5e5ea')
    fig.update_yaxes(title_text='Points', gridcolor='#e5e5ea', secondary_y=False)
    fig.update_yaxes(title_text='Rank', gridcolor='#e5e5ea', secondary_y=True, autorange='reversed')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ── Team Evolution (Set & Forget Comparison) ──
    render_team_evolution(team_id, gw_history, fetcher, players_df)
    
    st.markdown("---")
    
    # ── Best/Worst GWs ──
    st.markdown("### Highlights")
    
    hl_cols = st.columns(2)
    
    with hl_cols[0]:
        st.markdown("**Best Gameweek**")
        if best_gw:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #22c55e;border-radius:10px;padding:1rem;">
                <div style="color:#6ee7b7;font-size:0.72rem;font-weight:500;text-transform:uppercase;">GW{best_gw.get('event', '?')}</div>
                <div style="color:#22c55e;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{best_gw.get('points', 0)} pts</div>
                <div style="color:#86868b;font-size:0.8rem;">Rank: {best_gw.get('rank', 'N/A'):,}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with hl_cols[1]:
        st.markdown("**Worst Gameweek**")
        if worst_gw:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #ef4444;border-radius:10px;padding:1rem;">
                <div style="color:#fca5a5;font-size:0.72rem;font-weight:500;text-transform:uppercase;">GW{worst_gw.get('event', '?')}</div>
                <div style="color:#ef4444;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{worst_gw.get('points', 0)} pts</div>
                <div style="color:#86868b;font-size:0.8rem;">Rank: {worst_gw.get('rank', 'N/A'):,}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Transfer Analysis ──
    st.markdown("### Transfer Analysis")
    st.caption("Tracking free transfers and hits based on squad changes")
    
    # Attempt manual transfer detection if API data seems missing or for higher accuracy
    def get_manual_transfers(team_id, gw_history, fetcher):
        current_gw = fetcher.get_current_gameweek()
        all_gws = [gw.get('event') for gw in gw_history]
        
        manual_data = []
        prev_squad = set()
        banked = 1 # Start with 1 FT in GW1
        
        for i, gw in enumerate(all_gws):
            try:
                picks = fetcher.get_team_picks(team_id, gw)
                if isinstance(picks, dict) and 'picks' in picks:
                    current_squad = {p['element'] for p in picks['picks']}
                elif isinstance(picks, list):
                    current_squad = {p.get('element', p) for p in picks}
                else:
                    current_squad = set()
                
                if not prev_squad:
                    # GW1: No transfers possible
                    transfers_made = 0
                    free_used = 0
                    hits = 0
                else:
                    transfers_made = len(current_squad - prev_squad)
                    # 2024/25 rules: bank up to 5
                    if i > 0:
                        banked = min(5, banked + 1)
                    
                    free_used = min(transfers_made, banked)
                    hits = max(0, transfers_made - banked)
                    banked = max(0, banked - transfers_made)
                
                manual_data.append({
                    'gw': gw,
                    'total': transfers_made,
                    'free': free_used,
                    'hits': hits
                })
                prev_squad = current_squad
            except:
                # Fallback if specific GW fetch fails
                manual_data.append({
                    'gw': gw,
                    'total': gw_history[i].get('event_transfers', 0),
                    'free': max(0, gw_history[i].get('event_transfers', 0) - gw_history[i].get('event_transfers_cost', 0)//4),
                    'hits': gw_history[i].get('event_transfers_cost', 0)//4
                })
        return manual_data

    # Use session state to avoid repeated heavy API calls
    if f"manual_transfers_{team_id}" not in st.session_state:
        with st.spinner("Analyzing squad history..."):
            st.session_state[f"manual_transfers_{team_id}"] = get_manual_transfers(team_id, gw_history, fetcher)
    
    m_data = st.session_state[f"manual_transfers_{team_id}"]
    free_transfers = [d['free'] for d in m_data]
    hits_paid = [d['hits'] for d in m_data]
    
    fig2 = go.Figure()
    
    # Free Transfers (Green)
    fig2.add_trace(go.Bar(
        x=gws, y=free_transfers, name='Free Transfers',
        marker_color='#22c55e', opacity=0.8
    ))
    
    # Paid/Hit Transfers (Red)
    fig2.add_trace(go.Bar(
        x=gws, y=hits_paid, name='Paid Hits',
        marker_color='#ef4444', opacity=0.8
    ))
    
    fig2.update_layout(
        height=300,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(title='Gameweek', gridcolor='#e5e5ea'),
        yaxis=dict(title='Transfers', gridcolor='#e5e5ea', dtick=1),
        margin=dict(l=50, r=30, t=40, b=50)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Transfer ROI analysis
    st.markdown("**Transfer Hit ROI**")
    
    gws_with_hits = [(gw, gw.get('event_transfers_cost', 0), gw.get('points', 0)) 
                     for gw in gw_history if gw.get('event_transfers_cost', 0) > 0]
    
    if gws_with_hits:
        hit_df = pd.DataFrame([{
            'GW': gw.get('event', '?'),
            'Hit Cost': cost,
            'Points Scored': pts,
            'Net Points': pts - cost,
            'Worth It?': 'Yes' if pts > avg_gw_points + cost else 'No'
        } for gw, cost, pts in gws_with_hits])
        
        st.dataframe(hit_df, hide_index=True, use_container_width=True)
        
        hits_worth_it = sum(1 for gw, cost, pts in gws_with_hits if pts > avg_gw_points + cost)
        st.caption(f"Hits worth it: {hits_worth_it}/{len(gws_with_hits)} ({hits_worth_it/len(gws_with_hits)*100:.0f}%)")
    else:
        st.success("No transfer hits taken this season!")
    
    st.markdown("---")
    
    # ── Rank Movement ──
    st.markdown("### Rank Movement")
    
    rank_changes = []
    for i in range(1, len(gw_history)):
        prev_rank = gw_history[i-1].get('overall_rank', 0)
        curr_rank = gw_history[i].get('overall_rank', 0)
        if prev_rank > 0 and curr_rank > 0:
            change = prev_rank - curr_rank  # Positive = improvement
            rank_changes.append({
                'gw': gw_history[i].get('event', i+1),
                'change': change,
                'pts': gw_history[i].get('points', 0)
            })
    
    if rank_changes:
        green_arrows = sum(1 for rc in rank_changes if rc['change'] > 0)
        red_arrows = sum(1 for rc in rank_changes if rc['change'] < 0)
        
        arrow_cols = st.columns(2)
        with arrow_cols[0]:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #22c55e;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#22c55e;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{green_arrows}</div>
                <div style="color:#86868b;font-size:0.72rem;text-transform:uppercase;">Green Arrows</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with arrow_cols[1]:
            st.markdown(f'''
            <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #ef4444;border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#ef4444;font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{red_arrows}</div>
                <div style="color:#86868b;font-size:0.72rem;text-transform:uppercase;">Red Arrows</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
    # Always show league context at the bottom
    render_general_analysis(processor, players_df, fetcher)
        

def render_team_evolution(team_id, gw_history, fetcher, players_df):
    """Analyze how previous team versions would have performed if unchanged."""
    st.markdown("### Team Evolution Analysis")
    st.caption("Compare your actual performance vs 'frozen' versions of your squad (Set & Forget)")
    
    if not gw_history:
        return

    # Identify unique versions based on picks
    versions = []
    last_picks_set = set()
    
    current_gw = fetcher.get_current_gameweek()
    
    # We need player information to identify captains/subs correctly if we wanted full accuracy,
    # but for "Team Evolution", we'll use the 11 starters + captain bonus of that specific GW.
    # To keep it simple and fast, we'll use the starting 11 IDs.
    
    with st.spinner("Analyzing squad evolution history..."):
        all_gw_picks = {}
        for gw in gw_history:
            num = gw.get('event')
            if not num: continue
            
            picks_data = fetcher.get_team_picks(team_id, num)
            if not picks_data or 'picks' not in picks_data: continue
            
            picks = picks_data['picks']
            # Starters only (position 1-11)
            starters = [p['element'] for p in picks if p['position'] <= 11]
            captain = next((p['element'] for p in picks if p['is_captain']), None)
            
            picks_set = set(starters)
            
            if picks_set != last_picks_set:
                versions.append({
                    'gw': num,
                    'starters': starters,
                    'captain': captain,
                    'label': f"GW{num} Squad"
                })
                last_picks_set = picks_set
            
            all_gw_picks[num] = {
                'starters': starters,
                'captain': captain
            }

        if not versions:
            st.info("No squad changes detected yet.")
            return

        # Fetch live data (player points) for all GWs analyzed
        gw_points_map = {} # {gw: {player_id: points}}
        for gw in gw_history:
            num = gw.get('event')
            live_data = fetcher.get_event_live(num)
            elements = live_data.get('elements', [])
            player_scores = {e['id']: e['stats']['total_points'] for e in elements}
            gw_points_map[num] = player_scores

    # Calculate points for each version in each GW
    evolution_data = []
    
    for gw_entry in gw_history:
        num = gw_entry.get('event')
        actual_pts = gw_entry.get('points', 0)
        
        row = {'GW': num, 'Actual': actual_pts}
        v_points = []
        v_labels = []
        
        for v in versions:
            if num < v['gw']:
                row[v['label']] = None
            else:
                # Calculate points for this frozen squad in this GW
                v_starters = v['starters']
                v_captain = v['captain']
                
                gw_scores = gw_points_map.get(num, {})
                v_pts = sum(gw_scores.get(pid, 0) for pid in v_starters)
                cap_points = gw_scores.get(v_captain, 0)
                v_pts += cap_points 
                
                row[v['label']] = v_pts
                v_points.append(v_pts)
                v_labels.append(v['label'])
        
        # Calculate Hindsight Best for this GW
        # Include current actual performance (pre-hits) as a version to ensure 
        # Ultimate Team >= Actual (and accounts for chips/subs)
        hits = gw_entry.get('event_transfers_cost', 0)
        pre_hit_actual = actual_pts + hits
        v_points.append(pre_hit_actual)
        v_labels.append("Actual (Pre-Hits)")

        if v_points:
            max_p = max(v_points)
            best_idx = v_points.index(max_p)
            row['Hindsight Best'] = max_p
            source = v_labels[best_idx].replace(" Squad", "")
            row['Best Source'] = source if source != "Actual (Pre-Hits)" else "Actual"
        else:
            row['Hindsight Best'] = actual_pts
            row['Best Source'] = "Actual"

        evolution_data.append(row)

    df_evo = pd.DataFrame(evolution_data)
    
    # Chart
    fig = go.Figure()
    
    # Hindsight Best (Hindsight Optimal) - Fixed background line
    fig.add_trace(go.Scatter(
        x=df_evo['GW'], y=df_evo['Hindsight Best'], 
        name='Ultimate Team (Optimal)', 
        line=dict(color='rgba(139, 92, 246, 0.4)', width=4, dash='dot'),
        mode='lines+text+markers',
        text=df_evo['Best Source'],
        textposition="top center",
        textfont=dict(size=9, color='rgba(139, 92, 246, 0.8)'),
        hoverinfo='skip',
        showlegend=True
    ))

    # Actual performance
    fig.add_trace(go.Scatter(
        x=df_evo['GW'], y=df_evo['Actual'], 
        name='Actual Points', 
        line=dict(color='#3b82f6', width=3),
        mode='lines+markers'
    ))
    
    # Version filters
    selected_versions = st.multiselect(
        "Compare with versions from (Max 3):",
        options=[v['label'] for v in versions],
        default=[v['label'] for v in versions if v['gw'] == versions[0]['gw']][:3], # Default to first version, capped
        max_selections=3
    )
    
    colors = ['#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#10b981']
    
    for i, label in enumerate(selected_versions):
        if label in df_evo.columns:
            fig.add_trace(go.Scatter(
                x=df_evo['GW'], y=df_evo[label], 
                name=label,
                line=dict(color=colors[i % len(colors)], dash='dot'),
                mode='lines'
            ))
            
    fig.update_layout(
        height=400,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=30, t=40, b=50),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text='Gameweek', gridcolor='#e5e5ea')
    fig.update_yaxes(title_text='Points', gridcolor='#e5e5ea')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    if not df_evo.empty:
        latest_gw = df_evo['GW'].max()
        actual_total = df_evo['Actual'].sum()
        hindsight_total = df_evo['Hindsight Best'].sum()
        
        st.markdown("**Version Efficiency**")
        
        # Performance summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Actual Total", f"{int(actual_total)} pts")
        with m2:
            st.metric("Ultimate Team", f"{int(hindsight_total)} pts", f"{int(actual_total - hindsight_total)} pts", delta_color="inverse")
        with m3:
            gap = hindsight_total - actual_total
            efficiency = (actual_total / hindsight_total * 100) if hindsight_total > 0 else 100
            st.metric("Hindsight Efficiency", f"{efficiency:.1f}%", f"{gap:.0f} pts mismatch")

        st.markdown("---")
        st.markdown("**Specific Version Comparison**")
        
        if len(selected_versions) > 0:
            cols = st.columns(len(selected_versions))
            for i, label in enumerate(selected_versions):
                v_total = df_evo[label].sum()
                diff = actual_total - v_total
                with cols[i]:
                    delta_color = "normal" if diff >= 0 else "inverse"
                    st.metric(label, f"{int(v_total)} pts", f"{int(diff):+d} vs Actual", delta_color=delta_color)

        # ── Consolidated Squad Overlap Analysis ──
        st.markdown("---")
        st.markdown("**Squad Evolution Summary (Unique vs Common)**")
        
        if selected_versions:
            # Get latest num for actual squad reference
            latest_num = gw_history[-1].get('event')
            actual_picks = fetcher.get_team_picks(team_id, latest_num).get('picks', [])
            actual_starters = [p['element'] for p in actual_picks if p['position'] <= 11]
            actual_starters_set = set(actual_starters)
            
            # Prepare player name helper
            def get_names(ids):
                names = [players_df[players_df['id'] == pid]['web_name'].iloc[0] if pid in players_df['id'].values else f"ID:{pid}" for pid in ids]
                return ", ".join(sorted(names)) if names else "None"

            # Track sets for all versions
            version_sets = {}
            for label in selected_versions:
                v = next((ver for ver in versions if ver['label'] == label), None)
                if v:
                    version_sets[label] = set(v['starters'])
            
            # Common to ALL (Actual + selected versions)
            all_sets = [actual_starters_set] + list(version_sets.values())
            common_all_ids = set.intersection(*all_sets)
            
            # Construct consolidated table data (Wide Format)
            overlap_data = {
                "Retained (All)": [get_names(common_all_ids)]
            }
            
            for label in selected_versions:
                v_set = version_sets[label]
                trans_in = actual_starters_set - v_set
                trans_out = v_set - actual_starters_set
                
                overlap_data[f"{label} In"] = [get_names(trans_in)]
                overlap_data[f"{label} Out"] = [get_names(trans_out)]
            
            st.dataframe(pd.DataFrame(overlap_data), hide_index=True, use_container_width=True)
        else:
            st.info("Select squad versions above to see evolution analysis.")


def render_general_analysis(processor, players_df: pd.DataFrame, fetcher):
    """Render general analysis in expanders."""
    
    st.markdown("### League & Market Context")
    st.caption("General season trends and top performers")
    
    df = players_df.copy()
    df['total_points'] = safe_numeric(df.get('total_points', pd.Series([0]*len(df))))
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    # Top scorers this season
    with st.expander("Top Season Scorers", expanded=False):
        top_scorers = df.nlargest(15, 'total_points')[
            ['web_name', 'team_name', 'position', 'now_cost', 'total_points', 'selected_by_percent']
        ].copy()
        top_scorers.columns = ['Player', 'Team', 'Pos', 'Price', 'Total Points', 'EO%']
        
        st.dataframe(
            top_scorers.style.format({
                'Price': '£{:.1f}m',
                'Total Points': '{:.0f}',
                'EO%': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True,
            height=300
        )
        
        # Timeline graph for top 5 players
        st.markdown("**Top 5 Player Performance Timeline**")
        top_5_ids = df.nlargest(5, 'total_points')['id'].tolist()
        
        timeline_data = []
        for pid in top_5_ids:
            p_history = fetcher.get_player_summary(pid).get('history', [])
            p_name = df[df['id'] == pid].iloc[0]['web_name']
            
            cumulative = 0
            for gw_data in p_history:
                cumulative += gw_data.get('total_points', 0)
                timeline_data.append({
                    'GW': gw_data.get('round', 0),
                    'Points': cumulative,
                    'Player': p_name
                })
        
        if timeline_data:
            tdf = pd.DataFrame(timeline_data)
            fig_tl = go.Figure()
            
            for p_name in tdf['Player'].unique():
                pdf_p = tdf[tdf['Player'] == p_name]
                fig_tl.add_trace(go.Scatter(
                    x=pdf_p['GW'], y=pdf_p['Points'],
                    name=p_name, mode='lines+markers'
                ))
            
            fig_tl.update_layout(
                height=300,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(family='Inter, sans-serif', color='#86868b', size=11),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                xaxis=dict(title='Gameweek', gridcolor='#e5e5ea'),
                yaxis=dict(title='Cumulative Points', gridcolor='#e5e5ea'),
                margin=dict(l=40, r=20, t=30, b=40)
            )
            st.plotly_chart(fig_tl, use_container_width=True)
    
    # Points per million leaders
    with st.expander("Season Value Leaders (Pts/£m)", expanded=False):
        df['ppm'] = df['total_points'] / df['now_cost'].clip(lower=4)
        ppm_leaders = df[df['total_points'] > 30].nlargest(15, 'ppm')[
            ['web_name', 'team_name', 'position', 'now_cost', 'total_points', 'ppm']
        ].copy()
        ppm_leaders.columns = ['Player', 'Team', 'Pos', 'Price', 'Total Pts', 'Pts/£m']
        
        st.dataframe(
            ppm_leaders.style.format({
                'Price': '£{:.1f}m',
                'Total Pts': '{:.0f}',
                'Pts/£m': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True,
            height=450
        )
