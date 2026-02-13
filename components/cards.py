"""Card components for FPL Strategy Engine."""

import streamlit as st
import pandas as pd
from typing import Dict

from utils.helpers import (
    safe_numeric,
    get_player_fixtures,
    get_injury_status,
    get_price_change_info,
    FDR_COLORS
)


def render_player_detail_card(player_row: Dict, processor, players_df: pd.DataFrame):
    """Render detailed player card when searched."""
    player_id = player_row['id']
    
    # Header with name and basic info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    injury = get_injury_status(player_row)
    price_info = get_price_change_info(player_row)
    
    with col1:
        status_badge = f'<span style="color:{injury["color"]};margin-left:0.5rem;">[{injury["status"].upper()}]</span>' if injury['status'] != 'available' else ''
        st.markdown(f'''
        <div style="background:#1a1a1a;padding:1rem;border:1px solid #333;">
            <div style="font-size:1.5rem;font-weight:600;color:#fff;">
                {player_row["web_name"]}{status_badge}
            </div>
            <div style="color:#888;">{player_row.get("team_name", "")} | {player_row.get("position", "")} | {player_row["now_cost"]:.1f}m</div>
            {f'<div style="color:#f59e0b;font-size:0.85rem;margin-top:0.5rem;">{injury["news"]}</div>' if injury["news"] else ''}
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        ep = safe_numeric(pd.Series([player_row.get('ep_next', player_row.get('expected_points', 0))])).iloc[0]
        form = safe_numeric(pd.Series([player_row.get('form', 0)])).iloc[0]
        st.metric("Expected Points", f"{ep:.1f}")
        st.metric("Form", f"{form:.1f}")
    
    with col3:
        own = safe_numeric(pd.Series([player_row.get('selected_by_percent', 0)])).iloc[0]
        pts = safe_numeric(pd.Series([player_row.get('total_points', 0)])).iloc[0]
        st.metric("Ownership", f"{own:.1f}%")
        st.metric("Total Points", f"{int(pts)}")
    
    # Fixtures
    st.markdown('<p class="section-title">Next 5 Fixtures</p>', unsafe_allow_html=True)
    
    fixtures = get_player_fixtures(player_id, processor, 5)
    
    if fixtures:
        fix_cols = st.columns(5)
        
        for i, fix in enumerate(fixtures):
            with fix_cols[i]:
                venue = 'H' if fix['is_home'] else 'A'
                fdr = fix['fdr']
                color = FDR_COLORS.get(fdr, '#888')
                st.markdown(f'''
                <div style="background:{color};padding:0.5rem;text-align:center;border-radius:4px;">
                    <div style="font-weight:600;color:#fff;">GW{fix["gw"]}</div>
                    <div style="font-size:1.1rem;color:#fff;">{fix["opponent"]} ({venue})</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.7);">FDR: {fdr}</div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No upcoming fixtures found")
    
    # Price change info
    prc1, prc2 = st.columns(2)
    with prc1:
        st.markdown('<p class="section-title">Price Changes</p>', unsafe_allow_html=True)
        st.markdown(f'''
        <div style="background:#1a1a1a;padding:1rem;border:1px solid #333;">
            <div>This GW: <span style="color:{"#22c55e" if price_info["change_gw"] > 0 else "#ef4444" if price_info["change_gw"] < 0 else "#888"};font-weight:600;">{price_info["change_gw"]:+.1f}m</span></div>
            <div>Season: <span style="color:{"#22c55e" if price_info["change_season"] > 0 else "#ef4444" if price_info["change_season"] < 0 else "#888"};font-weight:600;">{price_info["change_season"]:+.1f}m</span></div>
            <div style="margin-top:0.5rem;">Transfers In: <span style="color:#22c55e;">{price_info["transfers_in"]:,}</span></div>
            <div>Transfers Out: <span style="color:#ef4444;">{price_info["transfers_out"]:,}</span></div>
            <div style="margin-top:0.5rem;">Net: <span style="color:{price_info["pred_color"]};font-weight:600;">{price_info["net_transfers"]:+,}</span></div>
            <div style="font-size:0.85rem;color:{price_info["pred_color"]};">Prediction: {price_info["prediction"].replace("_", " ").title()}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with prc2:
        st.markdown('<p class="section-title">Season Stats</p>', unsafe_allow_html=True)
        goals = int(safe_numeric(pd.Series([player_row.get('goals_scored', 0)])).iloc[0])
        assists = int(safe_numeric(pd.Series([player_row.get('assists', 0)])).iloc[0])
        cs = int(safe_numeric(pd.Series([player_row.get('clean_sheets', 0)])).iloc[0])
        mins = int(safe_numeric(pd.Series([player_row.get('minutes', 0)])).iloc[0])
        bonus = int(safe_numeric(pd.Series([player_row.get('bonus', 0)])).iloc[0])
        
        st.markdown(f'''
        <div style="background:#1a1a1a;padding:1rem;border:1px solid #333;">
            <div>Goals: <span style="color:#fff;font-weight:600;">{goals}</span></div>
            <div>Assists: <span style="color:#fff;font-weight:600;">{assists}</span></div>
            <div>Clean Sheets: <span style="color:#fff;font-weight:600;">{cs}</span></div>
            <div>Minutes: <span style="color:#fff;font-weight:600;">{mins:,}</span></div>
            <div>Bonus: <span style="color:#fff;font-weight:600;">{bonus}</span></div>
        </div>
        ''', unsafe_allow_html=True)


def render_captain_pick_card(player_row: Dict, rank: int = 1):
    """Render captain pick card."""
    ep = safe_numeric(pd.Series([player_row.get('expected_points', player_row.get('ep_next', 0))])).iloc[0]
    form = safe_numeric(pd.Series([player_row.get('form', 0)])).iloc[0]
    
    medal = '#1' if rank == 1 else '#2' if rank == 2 else '#3' if rank == 3 else f'#{rank}'
    
    st.markdown(f'''
    <div style="background:#1a1a1a;padding:0.75rem;border:1px solid #333;margin-bottom:0.5rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.2rem;margin-right:0.5rem;">{medal}</span>
                <span style="color:#fff;font-weight:600;">{player_row["web_name"]}</span>
                <span style="color:#888;margin-left:0.5rem;">{player_row.get("team_name", "")}</span>
            </div>
            <div style="text-align:right;">
                <div style="color:#22c55e;font-weight:600;">EP: {ep:.1f}</div>
                <div style="color:#888;font-size:0.85rem;">Form: {form:.1f}</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_injury_alert_card(player_row: Dict, injury_info: Dict):
    """Render injury alert card."""
    st.markdown(f'''
    <div style="background:#1a1a1a;padding:0.75rem;border-left:3px solid {injury_info["color"]};margin-bottom:0.5rem;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span style="color:#fff;font-weight:600;">{player_row["web_name"]}</span>
                <span style="color:#888;margin-left:0.5rem;">{player_row.get("team_name", "")}</span>
            </div>
            <span style="color:{injury_info["color"]};font-weight:600;">{injury_info["chance"]}%</span>
        </div>
        <div style="color:#888;font-size:0.85rem;margin-top:0.25rem;">{injury_info["news"]}</div>
    </div>
    ''', unsafe_allow_html=True)
