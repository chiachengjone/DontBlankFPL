"""Price Change Predictor Tab - Predict tonight's price rises/falls."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import (
    safe_numeric, style_df_with_injuries, normalize_name,
    calculate_consensus_ep, get_consensus_label
)
from components.styles import render_section_title


def render_price_predictor_tab(processor, players_df: pd.DataFrame):
    """Price Change Predictor tab - predict price rises and falls."""
    
    st.markdown('<p class="section-title">Price Change Predictor</p>', unsafe_allow_html=True)
    st.caption("Track transfer activity and predict price changes based on net transfers")
    
    # Metrics explanation dropdown
    with st.expander("Understanding Price Changes"):
        st.markdown("""
        **How FPL Prices Change**
        - Prices change at ~2:30 AM UK time based on transfer activity
        - Not exact: FPL uses a hidden algorithm to determine changes
        - Generally: ~60-80k net transfers needed for a price change
        
        **Transfer Metrics**
        - **Net Transfers**: Transfers In - Transfers Out this GW
        - **Cost Change Event**: Price change this GW so far (£0.1 increments)
        - **Cost Change Start**: Total price change from starting price
        
        **Prediction Categories**
        - **Likely to Rise** (green): Net transfers > 60k, buy before rise
        - **Likely to Fall** (red): Net transfers < -60k, sell before fall
        - **Watch List**: Close to threshold, monitor closely
        
        **Price Strategy Tips**
        - Buy rising players early in GW to bank value
        - Sell falling players before deadline
        - Don't chase price rises for players you'll sell soon
        - Hold quality players even if they're falling
        
        **Ownership Impact**
        - High ownership: Harder to rise, easier to fall
        - Low ownership: Easier to rise on small transfer volumes
        """)
    
    # Prepare data
    df = players_df.copy()
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    df = calculate_consensus_ep(df, active_models)
    con_label = get_consensus_label(active_models)
    
    df['transfers_in_event'] = safe_numeric(df.get('transfers_in_event', pd.Series([0]*len(df))))
    df['transfers_out_event'] = safe_numeric(df.get('transfers_out_event', pd.Series([0]*len(df))))
    df['transfers_in'] = safe_numeric(df.get('transfers_in', pd.Series([0]*len(df))))
    df['transfers_out'] = safe_numeric(df.get('transfers_out', pd.Series([0]*len(df))))
    df['net_transfers'] = df['transfers_in_event'] - df['transfers_out_event']
    df['net_transfers_total'] = df['transfers_in'] - df['transfers_out']
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['cost_change_event'] = safe_numeric(df.get('cost_change_event', pd.Series([0]*len(df))))
    df['cost_change_start'] = safe_numeric(df.get('cost_change_start', pd.Series([0]*len(df))))
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    
    # Price change thresholds (approximate)
    # Generally: ~80k net transfers for a price change, varies by ownership
    RISE_THRESHOLD = 60000
    FALL_THRESHOLD = -60000
    WATCH_THRESHOLD = 40000
    
    # ── Summary Metrics ──
    rising = df[df['net_transfers'] > RISE_THRESHOLD]
    falling = df[df['net_transfers'] < FALL_THRESHOLD]
    watch_rise = df[(df['net_transfers'] > WATCH_THRESHOLD) & (df['net_transfers'] <= RISE_THRESHOLD)]
    watch_fall = df[(df['net_transfers'] < -WATCH_THRESHOLD) & (df['net_transfers'] >= FALL_THRESHOLD)]
    
    sum_cols = st.columns(4)
    
    with sum_cols[0]:
        st.markdown(f'''
        <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #22c55e;border-radius:10px;padding:1rem;text-align:center;">
            <div style="color:#6ee7b7;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Likely to Rise</div>
            <div style="color:#22c55e;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{len(rising)}</div>
            <div style="color:#86868b;font-size:0.75rem;">players</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[1]:
        st.markdown(f'''
        <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #ef4444;border-radius:10px;padding:1rem;text-align:center;">
            <div style="color:#fca5a5;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Likely to Fall</div>
            <div style="color:#ef4444;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{len(falling)}</div>
            <div style="color:#86868b;font-size:0.75rem;">players</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[2]:
        st.markdown(f'''
        <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #4ade80;border-radius:10px;padding:1rem;text-align:center;">
            <div style="color:#bef264;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Watch (Rise)</div>
            <div style="color:#4ade80;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{len(watch_rise)}</div>
            <div style="color:#86868b;font-size:0.75rem;">near threshold</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with sum_cols[3]:
        st.markdown(f'''
        <div style="background:#ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #f59e0b;border-radius:10px;padding:1rem;text-align:center;">
            <div style="color:#fcd34d;font-size:0.72rem;font-weight:500;text-transform:uppercase;">Watch (Fall)</div>
            <div style="color:#f59e0b;font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{len(watch_fall)}</div>
            <div style="color:#86868b;font-size:0.75rem;">near threshold</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Price Rise Predictions ──
    st.markdown("### Expected Price Rises Tonight")
    st.caption("Players with net transfers above threshold - buy before they rise!")
    
    if not rising.empty:
        rise_df = rising.nlargest(15, 'net_transfers')[
            ['web_name', 'team_name', 'position', 'now_cost', 'net_transfers', 
             'transfers_in_event', 'consensus_ep', 'selected_by_percent']
        ].copy()
        rise_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Net Transfers', 'In', con_label, 'EO%']
        
        st.dataframe(
            rise_df.style.format({
                'Price': '£{:.1f}m',
                'Net Transfers': '{:+,.0f}',
                'In': '{:,.0f}',
                con_label: '{:.2f}',
                'EO%': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True,
            height=400
        )
    else:
        st.info("No players currently on track to rise tonight")
    
    st.markdown("---")
    
    # ── Price Fall Predictions ──
    st.markdown("### Expected Price Falls Tonight")
    st.caption("Players with heavy selling - consider avoiding or selling before they fall")
    
    if not falling.empty:
        fall_df = falling.nsmallest(15, 'net_transfers')[
            ['web_name', 'team_name', 'position', 'now_cost', 'net_transfers',
             'transfers_out_event', 'consensus_ep', 'selected_by_percent']
        ].copy()
        fall_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Net Transfers', 'Out', con_label, 'EO%']
        
        st.dataframe(
            fall_df.style.format({
                'Price': '£{:.1f}m',
                'Net Transfers': '{:+,.0f}',
                'Out': '{:,.0f}',
                con_label: '{:.2f}',
                'EO%': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True,
            height=400
        )
    else:
        st.info("No players currently on track to fall tonight")
    
    st.markdown("---")
    
    # ── Watch List ──
    st.markdown("### Price Change Watch List")
    st.caption("Players approaching thresholds - monitor closely")
    
    watch_tabs = st.tabs(["Watch (Rise)", "Watch (Fall)"])
    
    with watch_tabs[0]:
        if not watch_rise.empty:
            watch_r_df = watch_rise.nlargest(10, 'net_transfers')[
                ['web_name', 'team_name', 'position', 'now_cost', 'net_transfers', 'consensus_ep']
            ].copy()
            watch_r_df['threshold_pct'] = (watch_r_df['net_transfers'] / RISE_THRESHOLD * 100).round(0)
            watch_r_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Net Transfers', con_label, '% to Rise']
            
            st.dataframe(
                watch_r_df.style.format({
                    'Price': '£{:.1f}m',
                    'Net Transfers': '{:+,.0f}',
                    con_label: '{:.2f}',
                    '% to Rise': '{:.0f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No players currently on the rise watch list")
    
    with watch_tabs[1]:
        if not watch_fall.empty:
            watch_f_df = watch_fall.nsmallest(10, 'net_transfers')[
                ['web_name', 'team_name', 'position', 'now_cost', 'net_transfers', 'consensus_ep']
            ].copy()
            watch_f_df['threshold_pct'] = (abs(watch_f_df['net_transfers']) / abs(FALL_THRESHOLD) * 100).round(0)
            watch_f_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Net Transfers', con_label, '% to Fall']
            
            st.dataframe(
                watch_f_df.style.format({
                    'Price': '£{:.1f}m',
                    'Net Transfers': '{:+,.0f}',
                    con_label: '{:.2f}',
                    '% to Fall': '{:.0f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No players currently on the fall watch list")
    
    st.markdown("---")
    
    # ── Player Search ──
    st.markdown("### Search Player")
    
    search = st.text_input("Search for a specific player", key="price_search")
    
    if search:
        search_norm = normalize_name(search.lower().strip())
        df['_name_norm'] = df['web_name'].apply(lambda x: normalize_name(str(x).lower()))
        matched = df[df['_name_norm'].str.contains(search_norm, na=False)]
        
        if not matched.empty:
            for _, player in matched.head(3).iterrows():
                net = player['net_transfers']
                status_color = '#22c55e' if net > RISE_THRESHOLD else '#ef4444' if net < FALL_THRESHOLD else '#f59e0b' if abs(net) > WATCH_THRESHOLD else '#6b6b6b'
                status_text = 'RISING' if net > RISE_THRESHOLD else 'FALLING' if net < FALL_THRESHOLD else 'WATCH' if abs(net) > WATCH_THRESHOLD else 'STABLE'
                
                # Price change this event and season
                event_change = player['cost_change_event'] / 10
                season_change = player['cost_change_start'] / 10
                
                st.markdown(f'''
                <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.04);border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);padding:1rem;margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:#1d1d1f;font-weight:700;font-size:1.1rem;">{player['web_name']}</span>
                            <span style="color:#86868b;font-size:0.9rem;margin-left:0.5rem;">{player.get('team_name', '')} | {player['position']}</span>
                        </div>
                        <div style="background:{status_color};color:#fff;padding:0.25rem 0.75rem;border-radius:8px;font-size:0.72rem;font-weight:600;">
                            {status_text}
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-top:0.75rem;text-align:center;">
                        <div>
                            <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Current Price</div>
                            <div style="color:#1d1d1f;font-weight:600;font-family:'JetBrains Mono',monospace;">£{player['now_cost']:.1f}m</div>
                        </div>
                        <div>
                            <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Net Transfers</div>
                            <div style="color:{status_color};font-weight:600;font-family:'JetBrains Mono',monospace;">{net:+,.0f}</div>
                        </div>
                        <div>
                            <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">GW Change</div>
                            <div style="color:{'#22c55e' if event_change > 0 else '#ef4444' if event_change < 0 else '#6b6b6b'};font-weight:600;font-family:'JetBrains Mono',monospace;">{event_change:+.1f}</div>
                        </div>
                        <div>
                            <div style="color:#86868b;font-size:0.7rem;text-transform:uppercase;">Season Change</div>
                            <div style="color:{'#22c55e' if season_change > 0 else '#ef4444' if season_change < 0 else '#6b6b6b'};font-weight:600;font-family:'JetBrains Mono',monospace;">{season_change:+.1f}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("No players found matching your search")
    
    st.markdown("---")
    
    # ── Season Price Changes ──
    st.markdown("### Season Price Movers")
    
    mover_tabs = st.tabs(["Biggest Risers", "Biggest Fallers"])
    
    with mover_tabs[0]:
        risers = df.nlargest(12, 'cost_change_start')
        if not risers.empty:
            risers_df = risers[['web_name', 'team_name', 'position', 'now_cost', 'cost_change_start', 
                                'consensus_ep', 'selected_by_percent']].copy()
            risers_df['cost_change_start'] = risers_df['cost_change_start'] / 10
            risers_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Season +/-', con_label, 'EO%']
            
            st.dataframe(
                risers_df.style.format({
                    'Price': '£{:.1f}m',
                    'Season +/-': '+£{:.1f}m',
                    con_label: '{:.2f}',
                    'EO%': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    with mover_tabs[1]:
        fallers = df.nsmallest(12, 'cost_change_start')
        if not fallers.empty:
            fallers_df = fallers[['web_name', 'team_name', 'position', 'now_cost', 'cost_change_start',
                                  'consensus_ep', 'selected_by_percent']].copy()
            fallers_df['cost_change_start'] = fallers_df['cost_change_start'] / 10
            fallers_df.columns = ['Player', 'Team', 'Pos', 'Price', 'Season +/-', con_label, 'EO%']
            
            st.dataframe(
                fallers_df.style.format({
                    'Price': '£{:.1f}m',
                    'Season +/-': '{:.1f}m',
                    con_label: '{:.2f}',
                    'EO%': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    st.markdown("---")
    
    # ── Transfer Activity Chart ──
    st.markdown("### Transfer Activity Distribution")
    
    # Histogram of net transfers
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['net_transfers'],
        nbinsx=50,
        marker_color='#3b82f6',
        opacity=0.7
    ))
    
    # Add threshold lines
    fig.add_vline(x=RISE_THRESHOLD, line_dash="dash", line_color="#22c55e", 
                  annotation_text="Rise Threshold", annotation_position="top right")
    fig.add_vline(x=FALL_THRESHOLD, line_dash="dash", line_color="#ef4444",
                  annotation_text="Fall Threshold", annotation_position="top left")
    
    fig.update_layout(
        height=300,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='Inter, sans-serif', color='#86868b', size=11),
        xaxis=dict(title='Net Transfers This GW', gridcolor='#e5e5ea'),
        yaxis=dict(title='Number of Players', gridcolor='#e5e5ea'),
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
