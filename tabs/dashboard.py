"""Dashboard tab - At-a-glance overview for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import html

from config import (
    CAPTAIN_MULTIPLIER,
    DASHBOARD_TOP_PICKS,
    DASHBOARD_FIXTURE_SWINGS,
    POSITION_COLORS,
    FDR_COLOR_MAP,
    OWNERSHIP_TEMPLATE_THRESHOLD,
    OWNERSHIP_POPULAR_THRESHOLD,
    OWNERSHIP_DIFFERENTIAL_THRESHOLD,
    get_chart_layout,
)
from utils.helpers import (
    safe_numeric,
    get_injury_status,
    get_availability_badge,
    get_ownership_tier,
    classify_ownership_column,
    style_df_with_injuries,
    round_df,
    calculate_consensus_ep,
    get_consensus_label,
)


def render_dashboard_tab(processor, players_df: pd.DataFrame):
    """
    Dashboard Overview.
    
    This is the landing page of the application. It provides:
    1.  **Key Metrics**: Gameweek averages, your score, and active models.
    2.  **Top Picks**: The best players for the upcoming gameweek according to the selected models.
    3.  **Insights**: Captaincy recommendations, fixture swings, and market trends.
    """
    
    # ── Preparation ──
    # Work on a copy to avoid affecting other tabs
    players_df = players_df.copy()
    
    # If ML predictions have been run (in the background or via the ML tab),
    # we merge them here so the dashboard can use them.
    if 'ml_predictions' in st.session_state and 'ml_pred' not in players_df.columns:
        ml_preds = st.session_state['ml_predictions']
        players_df['ml_pred'] = players_df['id'].apply(
            lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else 0
        )

    # Calculate Consensus EP (Model xP)
    # This blends the selected models (ML, Poisson, FPL) into a single "Master Prediction".
    active_models = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    players_df = calculate_consensus_ep(players_df, active_models)
    con_label = get_consensus_label(active_models)

    # ── Help & Explanations ──
    with st.expander("Understanding Dashboard Metrics"):
        st.markdown(f"""
        **Model xP ({con_label})**
        This is your "Master Prediction". It's a weighted average of the models you've enabled:
        - **All 3 Active**: ML (40%), Poisson (40%), FPL (20%). *Balanced approach.*
        - **ML + Poisson**: 50/50 split. *Pure data-driven.*
        - **ML + FPL**: ML (67%), FPL (33%). *Smart but grounded.*
        - **Single Model**: 100% of that model.
        
        **Gameweek Overview**
        - **GW Average**: The average score of all FPL managers.
        - **vs Average**: How much you beat (or lost to) the average. Green is good!
        
        **Top Picks**
        - The highest predicted scorers for the next gameweek.
        - **Yellow Background**: The player has an injury flag (check before buying!).
        - **Tiers**: Tells you if a player is "Template" (everyone owns them) or a "Differential".
        
        **Fixture Swings**
        - **Green**: Schedule is getting easier. Target these teams.
        - **Red**: Schedule is getting harder. Avoid or sell.
        """)
        
    # ── Current Status ──
    try:
        current_gw = processor.fetcher.get_current_gameweek()
    except Exception:
        current_gw = 1

    # Fetch live score data (Average and User)
    gw_avg_score = 0
    user_gw_points = None
    team_id = st.session_state.get('fpl_team_id', 0)
    
    # 1. Get Global Average
    try:
        bootstrap = processor.fetcher.get_bootstrap_static()
        events = bootstrap.get('events', [])
        for ev in events:
            if ev.get('id') == current_gw:
                gw_avg_score = ev.get('average_entry_score', 0)
                break
    except Exception:
        pass

    # 2. Get Your Team's Score (if Team ID is set)
    if team_id and team_id > 0:
        try:
            picks_data = processor.fetcher.get_team_picks(team_id, current_gw)
            if isinstance(picks_data, dict):
                entry_hist = picks_data.get('entry_history', {})
                user_gw_points = entry_hist.get('points', None)
                # Adjust for hits taken
                transfer_cost = entry_hist.get('event_transfers_cost', 0)
                if user_gw_points is not None:
                    user_gw_points = user_gw_points - transfer_cost
        except Exception:
            pass

    # ── Gameweek Scoreboard ──
    st.markdown('<p class="section-title">Gameweek Overview</p>', unsafe_allow_html=True)
    
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Current GW", current_gw)
    with g2:
        st.metric("GW Average", gw_avg_score if gw_avg_score else "N/A")
    with g3:
        if user_gw_points is not None:
            st.metric("Your GW Score", user_gw_points)
        else:
            st.metric("Your Team ID", team_id if team_id else "Not set", help="Enter Team ID in settings above")
    with g4:
        if user_gw_points is not None and gw_avg_score:
            diff = user_gw_points - gw_avg_score
            color = '#22c55e' if diff >= 0 else '#ef4444'
            st.metric("vs Average", f"{diff:+d}", delta=f"{diff:+d}")
        else:
            avg_ep = safe_numeric(players_df.get('expected_points', players_df.get('ep_next', pd.Series([0] * len(players_df))))).mean()
            st.metric("Avg xP (all)", f"{avg_ep:.2f}")

    # ── Widget Grid ──
    render_top_picks_summary(players_df)
    render_captain_quick_pick(players_df)

    with st.container():
        render_squad_performance(processor, players_df, current_gw, team_id)

    with st.container():
        render_fixture_swings(processor)

    with st.container():
        render_ownership_breakdown(players_df)

    with st.container():
        render_transfer_trends(players_df)


# ── Section Renderers ──


def render_squad_performance(processor, players_df, current_gw, team_id):
    """Show whether the user's squad is over/underperforming expectations."""
    if not team_id or team_id <= 0:
        return

    try:
        picks_data = processor.fetcher.get_team_picks(team_id, current_gw)
        if not isinstance(picks_data, dict) or 'picks' not in picks_data:
            return

        picks = picks_data['picks']
        starting_ids = [
            p['element'] for p in picks
            if p.get('position', p.get('multiplier', 0)) <= 11 or p.get('multiplier', 0) > 0
        ][:11]
        captain_id = next((p['element'] for p in picks if p.get('is_captain', False)), None)

        squad_df = players_df[players_df['id'].isin(starting_ids)].copy()
        if squad_df.empty:
            return

        squad_df['total_points'] = safe_numeric(squad_df.get('total_points', pd.Series([0] * len(squad_df))))
        # Use Consensus EP (Model xP) calculated in render_dashboard_tab
        squad_df['model_xp'] = safe_numeric(squad_df['consensus_ep'])
        squad_df['minutes'] = safe_numeric(squad_df.get('minutes', pd.Series([0] * len(squad_df))))
        squad_df['form'] = safe_numeric(squad_df.get('form', pd.Series([0] * len(squad_df))))
        squad_df['games_played'] = (squad_df['minutes'] / 90).clip(lower=1)
        squad_df['pts_per_game'] = squad_df['total_points'] / squad_df['games_played']
        squad_df['expected_per_game'] = squad_df['model_xp']
        squad_df['perf_diff'] = squad_df['pts_per_game'] - squad_df['expected_per_game']

        st.markdown('<p class="section-title">Squad Performance</p>', unsafe_allow_html=True)
        st.caption("Comparing your starting XI's actual points-per-game vs expected - are you unlucky or riding form?")

        # Summary metrics
        avg_actual = squad_df['pts_per_game'].mean()
        avg_expected = squad_df['expected_per_game'].mean()
        perf_delta = avg_actual - avg_expected
        perf_pct = (perf_delta / max(avg_expected, 0.1)) * 100

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Avg Pts/Game", f"{avg_actual:.2f}")
        with p2:
            st.metric(f"{get_consensus_label(st.session_state.active_models)}/Game", f"{avg_expected:.2f}")
        with p3:
            label = "Overperforming" if perf_delta >= 0 else "Underperforming"
            st.metric(label, f"{perf_delta:+.2f} pts/gm", delta=f"{perf_pct:+.0f}%")
        with p4:
            overperf_count = (squad_df['perf_diff'] > 0).sum()
            st.metric(f"Players Above {get_consensus_label(st.session_state.active_models, 1)}", f"{overperf_count}/{len(squad_df)}")

        # Per-player bar chart
        chart_df = squad_df.sort_values('perf_diff', ascending=True).copy()
        chart_df['label'] = chart_df['web_name'].apply(lambda x: str(x) if pd.notna(x) else '')
        if captain_id and captain_id in chart_df['id'].values:
            chart_df.loc[chart_df['id'] == captain_id, 'label'] = chart_df.loc[chart_df['id'] == captain_id, 'label'] + ' (C)'

        colors = ['#22c55e' if d >= 0 else '#ef4444' for d in chart_df['perf_diff']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=chart_df['label'],
            x=chart_df['perf_diff'],
            orientation='h',
            marker_color=colors,
            text=chart_df['perf_diff'].apply(lambda v: f"{v:+.2f}"),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Actual: %{customdata[0]:.2f}/gm<br>xP: %{customdata[1]:.2f}/gm<extra></extra>',
            customdata=chart_df[['pts_per_game', 'expected_per_game']].values,
        ))
        fig.update_layout(
            height=max(280, len(chart_df) * 30),
            template='plotly_white',
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Inter, sans-serif', color='#86868b', size=11),
            xaxis=dict(title=f'Pts/Game vs {get_consensus_label(st.session_state.active_models)}', gridcolor='#e5e5ea', zeroline=True,
                      zerolinecolor='rgba(0,0,0,0.06)', zerolinewidth=1),
            yaxis=dict(gridcolor='#e5e5ea'),
            margin=dict(l=120, r=50, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch", key='dash_squad_perf')

    except Exception:
        pass


def render_top_picks_summary(players_df: pd.DataFrame):
    """Top xP picks per position - quick recommendations."""
    st.markdown('<p class="section-title">Top Picks by Position</p>', unsafe_allow_html=True)
    st.caption(f"Highest {get_consensus_label(st.session_state.active_models)} this gameweek per position")

    df = players_df.copy()
    # Use pre-calculated consensus_ep
    df['ep'] = safe_numeric(df['consensus_ep'])
    df['now_cost'] = safe_numeric(df['now_cost'], 5)
    df['selected_by_percent'] = safe_numeric(df['selected_by_percent'])
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0] * len(df))))
    df = df[df['minutes'] > 90]

    cols = st.columns(4)
    for i, pos in enumerate(['GKP', 'DEF', 'MID', 'FWD']):
        with cols[i]:
            pos_df = df[df['position'] == pos].nlargest(DASHBOARD_TOP_PICKS, 'ep')
            st.markdown(f"**{pos}**")
            if pos_df.empty:
                st.caption("No data")
                continue
            for _, p in pos_df.iterrows():
                injury = get_injury_status(p)
                # Highlight background yellow if injured
                is_injured = injury['status'] != 'available'
                bg_color = '#fff9e6' if is_injured else '#fff'
                
                tier = get_ownership_tier(safe_numeric(pd.Series([p['selected_by_percent']])).iloc[0])
                player_name = html.escape(str(p["web_name"]) if pd.notna(p["web_name"]) else "")
                team_name = html.escape(str(p.get("team_name", "")) if pd.notna(p.get("team_name")) else "")
                st.markdown(
                    f'<div style="background:{bg_color};border:1px solid rgba(0,0,0,0.06);border-radius:8px;'
                    f'padding:0.5rem 0.7rem;margin-bottom:0.3rem;">'
                    f'<div style="color:#1d1d1f;font-weight:600;font-size:0.85rem;">'
                    f'{player_name}</div>'
                    f'<div style="color:#888;font-size:0.72rem;">'
                    f'{team_name} | {p["now_cost"]:.1f}m | '
                    f'<span style="color:{POSITION_COLORS.get(pos, "#888")}">{get_consensus_label(st.session_state.active_models)} {p["ep"]:.2f}</span> | '
                    f'<span style="color:{tier["color"]}">{tier["tier"]}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


def render_captain_quick_pick(players_df: pd.DataFrame):
    """Top 3 captain picks."""
    st.markdown('<p class="section-title">Captain Quick Pick</p>', unsafe_allow_html=True)

    df = players_df.copy()
    from utils.helpers import calculate_enhanced_captain_score
    active_m = st.session_state.get('active_models', ['ml', 'poisson', 'fpl'])
    
    # Ensure necessary metrics exist for UI display
    df['ep'] = safe_numeric(df.get('consensus_ep', 0))
    df['form'] = safe_numeric(df.get('form', 0))
    df['minutes'] = safe_numeric(df.get('minutes', 0))
    
    # Calculate scores on players with sufficient minutes
    viable_df = df[df['minutes'] > 500].copy()
    if viable_df.empty:
        viable_df = df.copy()
        
    viable_df['captain_score'] = viable_df.apply(lambda r: calculate_enhanced_captain_score(r, active_m), axis=1)
    viable_df['captain_ev'] = viable_df['ep'] * CAPTAIN_MULTIPLIER

    top3 = viable_df.nlargest(3, 'captain_score')
    cap_cols = st.columns(3)

    for i, (_, cap) in enumerate(top3.iterrows()):
        with cap_cols[i]:
            badge = ['1ST CHOICE', '2ND CHOICE', '3RD CHOICE'][i]
            cap_name = html.escape(str(cap["web_name"]) if pd.notna(cap["web_name"]) else "")
            cap_team = html.escape(str(cap.get("team_name", "")) if pd.notna(cap.get("team_name")) else "")
            
            # Match the standard card aesthetic from the rest of the app
            bg_color = "#ffffff"
            border_color = "rgba(0,0,0,0.06)"
            text_color = "#1d1d1f"
            secondary_text = "#86868b"
            
            # Highlight top choice with a subtle border color or shadow
            card_shadow = "0 4px 12px rgba(0,0,0,0.06)"
            if i == 0:
                border_color = "#007AFF"
                card_shadow = "0 4px 15px rgba(0,122,255,0.1)"
                
            st.markdown(
                f'<div style="background:{bg_color}; border:1px solid {border_color}; border-radius:14px; padding:1.25rem 1rem; text-align:center; '
                f'box-shadow:{card_shadow}; transition:transform 0.2s ease; animation:fadeInUp 0.4s ease-out both;">'
                f'<div style="font-size:0.6rem; color:{secondary_text}; letter-spacing:0.1em; font-weight:700;">{badge}</div>'
                f'<div style="font-size:1.2rem; font-weight:700; color:{text_color}; margin:0.4rem 0;">{cap_name}</div>'
                f'<div style="color:{secondary_text}; font-size:0.75rem;">{cap_team} | £{cap["now_cost"]:.1f}m</div>'
                f'<div style="background:rgba(0,0,0,0.03); border-radius:8px; padding:0.4rem; margin-top:0.75rem;">'
                f'<div style="color:{text_color}; font-weight:700; font-size:1.1rem; font-family:\'JetBrains Mono\',monospace;">{cap["captain_ev"]:.2f} xP</div>'
                f'<div style="color:{secondary_text}; font-size:0.6rem; text-transform:uppercase;">Captain Expected Total</div>'
                f'</div>'
                f'<div style="color:{secondary_text}; font-size:0.7rem; margin-top:0.6rem; font-family:\'JetBrains Mono\',monospace;">'
                f'Form {cap["form"]:.2f} | Base {cap["ep"]:.2f}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )



def render_fixture_swings(processor):
    """Teams with biggest upcoming fixture difficulty changes."""
    st.markdown('<p class="section-title">Fixture Swings</p>', unsafe_allow_html=True)
    st.caption("Teams whose schedule is about to get easier or harder over the next 6 GWs")

    try:
        current_gw = processor.fetcher.get_current_gameweek()
        fixtures = processor.fixtures_df
        teams_df = processor.teams_df

        if fixtures is None or fixtures.empty:
            st.info("Fixture data unavailable")
            return

        fixtures = fixtures.copy()
        fixtures['event'] = safe_numeric(fixtures.get('event', pd.Series([0] * len(fixtures))))

        # Calculate near (next 3) vs far (GW+4 to GW+6) FDR
        near_range = list(range(current_gw + 1, current_gw + 4))
        far_range = list(range(current_gw + 4, current_gw + 7))

        swing_data = []
        for _, team in teams_df.iterrows():
            tid = team['id']
            near_fdr = _avg_fdr_for_range(fixtures, tid, near_range)
            far_fdr = _avg_fdr_for_range(fixtures, tid, far_range)
            swing = near_fdr - far_fdr  # positive = getting easier
            swing_data.append({
                'Team': team['short_name'],
                'Near FDR': round(near_fdr, 2),
                'Far FDR': round(far_fdr, 2),
                'Swing': round(swing, 2),
            })

        swing_df = pd.DataFrame(swing_data)
        if swing_df.empty:
            st.info("No swing data available")
            return

        # Show biggest improvers and decliners
        sw1, sw2 = st.columns(2)

        with sw1:
            st.markdown("**Getting Easier**")
            improving = swing_df.nlargest(DASHBOARD_FIXTURE_SWINGS, 'Swing')
            improving_display = improving.copy()
            # st.dataframe with column_config handles formatting while preserving sort order
            st.dataframe(
                improving_display, 
                hide_index=True, 
                width="stretch",
                column_config={
                    "Swing": st.column_config.NumberColumn(
                        "Swing",
                        format="%+.2f",
                        help="Change in fixture difficulty (Positive = Easier)"
                    )
                }
            )

        with sw2:
            st.markdown("**Getting Harder**")
            declining = swing_df.nsmallest(DASHBOARD_FIXTURE_SWINGS, 'Swing')
            declining_display = declining.copy()
            # declining_display['Swing'] is numeric
            st.dataframe(
                declining_display, 
                hide_index=True, 
                width="stretch",
                column_config={
                    "Swing": st.column_config.NumberColumn(
                        "Swing",
                        format="%.2f",  # Already negative, no need for + sign
                        help="Change in fixture difficulty (Negative = Harder)"
                    )
                }
            )

        # Bar chart
        sorted_swing = swing_df.sort_values('Swing', ascending=True)
        colors = ['#22c55e' if s > 0.1 else '#ef4444' if s < -0.1 else '#555' for s in sorted_swing['Swing']]
        fig = go.Figure(go.Bar(
            x=sorted_swing['Swing'],
            y=sorted_swing['Team'],
            orientation='h',
            marker_color=colors,
            text=sorted_swing['Swing'].apply(lambda x: f"{x:+.2f}"),
            textposition='outside',
        ))
        fig.add_vline(x=0, line_color='#86868b', line_dash='dot')
        fig.update_layout(**get_chart_layout(
            height=500,
            xaxis=dict(title='Fixture Swing (positive = getting easier)'),
            yaxis=dict(title=''),
            margin=dict(l=60, r=60, t=20, b=40),
        ))
        st.plotly_chart(fig, width="stretch", key='dashboard_fixture_swings_chart')

    except Exception as e:
        st.info(f"Fixture swing data unavailable: {e}")


def render_ownership_breakdown(players_df: pd.DataFrame):
    """Pie chart of ownership tiers across all players with >90 mins."""
    st.markdown('<p class="section-title">Ownership Landscape</p>', unsafe_allow_html=True)
    st.caption("Distribution of players by ownership tier (min 90 minutes played)")

    df = players_df.copy()
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0] * len(df))))
    df = df[df['minutes'] > 90]
    df['tier'] = classify_ownership_column(df)

    tier_counts = df['tier'].value_counts()
    tier_colors = {
        'Template': '#3b82f6',
        'Popular': '#22c55e',
        'Enabler': '#f59e0b',
        'Differential': '#a855f7',
    }

    o1, o2 = st.columns([1, 2])
    with o1:
        fig = go.Figure(go.Pie(
            labels=tier_counts.index.tolist(),
            values=tier_counts.values.tolist(),
            marker_colors=[tier_colors.get(t, '#888') for t in tier_counts.index],
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(color='#fff'),
        ))
        fig.update_layout(**get_chart_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=20, b=20)))
        st.plotly_chart(fig, width="stretch", key='dashboard_ownership_pie')

    with o2:
        st.markdown("**Tier Definitions**")
        st.markdown(
            f'- **Template** (>={int(OWNERSHIP_TEMPLATE_THRESHOLD)}%): '
            f'{tier_counts.get("Template", 0)} players -- core picks most managers own\n'
            f'- **Popular** (>={int(OWNERSHIP_POPULAR_THRESHOLD)}%): '
            f'{tier_counts.get("Popular", 0)} players -- well-owned mainstream picks\n'
            f'- **Enabler** (>={int(OWNERSHIP_DIFFERENTIAL_THRESHOLD)}%): '
            f'{tier_counts.get("Enabler", 0)} players -- moderate ownership\n'
            f'- **Differential** (<{int(OWNERSHIP_DIFFERENTIAL_THRESHOLD)}%): '
            f'{tier_counts.get("Differential", 0)} players -- low owned, high edge potential'
        )


def render_transfer_trends(players_df: pd.DataFrame):
    """Quick view of most transferred in/out."""
    st.markdown('<p class="section-title">Transfer Trends</p>', unsafe_allow_html=True)

    df = players_df.copy()
    df['transfers_in_event'] = safe_numeric(df.get('transfers_in_event', pd.Series([0] * len(df))))
    df['transfers_out_event'] = safe_numeric(df.get('transfers_out_event', pd.Series([0] * len(df))))
    df['net_transfers'] = df['transfers_in_event'] - df['transfers_out_event']
    df['minutes'] = safe_numeric(df.get('minutes', pd.Series([0] * len(df))))
    df = df[df['minutes'] > 0]

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Most Transferred In**")
        risers = df.nlargest(8, 'transfers_in_event')[['web_name', 'team_name', 'now_cost', 'transfers_in_event']].copy()
        risers.columns = ['Player', 'Team', 'Price', 'Transfers In']
        # risers['Transfers In'] is numeric
        st.dataframe(
            style_df_with_injuries(risers), 
            hide_index=True, 
            width="stretch",
            column_config={
                "Transfers In": st.column_config.NumberColumn(
                    "Transfers In",
                    format="+%d"
                ),
                "Price": st.column_config.NumberColumn("Price", format="£%.1fm")
            }
        )

    with t2:
        st.markdown("**Most Transferred Out**")
        fallers = df.nlargest(8, 'transfers_out_event')[['web_name', 'team_name', 'now_cost', 'transfers_out_event']].copy()
        fallers.columns = ['Player', 'Team', 'Price', 'Transfers Out']
        # fallers['Transfers Out'] is numeric
        st.dataframe(
            style_df_with_injuries(fallers), 
            hide_index=True, 
            width="stretch",
            column_config={
                "Transfers Out": st.column_config.NumberColumn(
                    "Transfers Out",
                    format="-%d"
                ),
                "Price": st.column_config.NumberColumn("Price", format="£%.1fm")
            }
        )


# ── Internal Helpers ──

def _avg_fdr_for_range(fixtures_df, team_id, gw_range):
    """Calculate average FDR for a team across a gameweek range."""
    total = 0
    count = 0
    for gw in gw_range:
        gw_fix = fixtures_df[fixtures_df['event'] == gw]
        home = gw_fix[gw_fix['team_h'] == team_id]
        away = gw_fix[gw_fix['team_a'] == team_id]
        for _, m in home.iterrows():
            total += safe_numeric(pd.Series([m.get('team_h_difficulty', 3)])).iloc[0]
            count += 1
        for _, m in away.iterrows():
            total += safe_numeric(pd.Series([m.get('team_a_difficulty', 3)])).iloc[0]
            count += 1
    return total / count if count > 0 else 3.0
