"""Helper utility functions for FPL Strategy Engine."""

import pandas as pd
import numpy as np
from typing import Dict, List

from config import (
    OWNERSHIP_TEMPLATE_THRESHOLD,
    OWNERSHIP_POPULAR_THRESHOLD,
    OWNERSHIP_DIFFERENTIAL_THRESHOLD,
    ROTATION_SAFE_MINUTES_PCT,
    ROTATION_MODERATE_PCT,
    UNAVAILABLE_STATUSES,
)


def safe_numeric(series, default=0):
    """Safely convert series to numeric, handling NaN values."""
    return pd.to_numeric(series, errors='coerce').fillna(default)


def get_player_name(player_id: int, players_df: pd.DataFrame) -> str:
    """Get player name from ID."""
    player = players_df[players_df['id'] == player_id]
    if not player.empty:
        return player.iloc[0]['web_name']
    return f"Unknown ({player_id})"


def get_team_short_name(team_id: int, processor) -> str:
    """Get team short name from ID."""
    try:
        teams = processor.teams_df
        team = teams[teams['id'] == team_id]
        if not team.empty:
            return team.iloc[0].get('short_name', team.iloc[0].get('name', 'UNK'))
        return 'UNK'
    except:
        return 'UNK'


def get_player_fixtures(player_id: int, processor, weeks_ahead: int = 5) -> List[Dict]:
    """Get player's next fixtures with difficulty ratings."""
    try:
        player_row = processor.players_df[processor.players_df['id'] == player_id]
        if player_row.empty:
            return []
        
        team_id = player_row.iloc[0]['team']
        current_gw = processor.fetcher.get_current_gameweek()
        fixtures_df = processor.fixtures_df
        
        fixtures = fixtures_df[
            (fixtures_df['event'] >= current_gw) &
            (fixtures_df['event'] < current_gw + weeks_ahead) &
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
        ].sort_values('event')
        
        result = []
        for _, fix in fixtures.iterrows():
            is_home = fix['team_h'] == team_id
            opponent_id = fix['team_a'] if is_home else fix['team_h']
            fdr = fix.get('team_h_difficulty', 3) if is_home else fix.get('team_a_difficulty', 3)
            
            result.append({
                'gw': int(fix['event']),
                'opponent': get_team_short_name(opponent_id, processor),
                'is_home': is_home,
                'fdr': int(fdr),
                'kickoff': fix.get('kickoff_time', '')
            })
        return result
    except:
        return []


def _safe_int(value, default=100):
    """Safely convert value to int, handling NaN/None."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def get_injury_status(player_row) -> Dict:
    """Get injury/availability status for a player."""
    news = player_row.get('news', '') or ''
    chance = player_row.get('chance_of_playing_next_round')
    status = player_row.get('status', 'a')
    
    # Convert chance to safe int
    chance_int = _safe_int(chance, 100)
    
    if status == 'i':
        return {'status': 'injured', 'color': '#ef4444', 'icon': 'X', 'chance': 0, 'news': news}
    elif status == 's':
        return {'status': 'suspended', 'color': '#ef4444', 'icon': 'S', 'chance': 0, 'news': news}
    elif status == 'u':
        return {'status': 'unavailable', 'color': '#ef4444', 'icon': '!', 'chance': 0, 'news': news}
    elif status == 'd':
        return {'status': 'doubtful', 'color': '#f59e0b', 'icon': '?', 'chance': chance_int if chance_int < 100 else 50, 'news': news}
    elif chance_int < 100:
        return {'status': 'doubt', 'color': '#f59e0b', 'icon': '?', 'chance': chance_int, 'news': news}
    else:
        return {'status': 'available', 'color': '#22c55e', 'icon': '', 'chance': 100, 'news': ''}


def get_price_change_info(player_row) -> Dict:
    """Get price change info for a player."""
    cost_change_event = safe_numeric(pd.Series([player_row.get('cost_change_event', 0)])).iloc[0]
    cost_change_start = safe_numeric(pd.Series([player_row.get('cost_change_start', 0)])).iloc[0]
    transfers_in = safe_numeric(pd.Series([player_row.get('transfers_in_event', 0)])).iloc[0]
    transfers_out = safe_numeric(pd.Series([player_row.get('transfers_out_event', 0)])).iloc[0]
    
    net_transfers = transfers_in - transfers_out
    
    # Simple price rise/fall prediction based on net transfers
    if net_transfers > 50000:
        prediction = 'likely_rise'
        pred_color = '#22c55e'
    elif net_transfers < -50000:
        prediction = 'likely_fall'
        pred_color = '#ef4444'
    else:
        prediction = 'stable'
        pred_color = '#888'
    
    return {
        'change_gw': cost_change_event / 10,
        'change_season': cost_change_start / 10,
        'transfers_in': int(transfers_in),
        'transfers_out': int(transfers_out),
        'net_transfers': int(net_transfers),
        'prediction': prediction,
        'pred_color': pred_color
    }


# FDR color mapping
FDR_COLORS = {
    1: '#22c55e',  # Green - easy
    2: '#77c45e',  # Light green
    3: '#f59e0b',  # Yellow - medium
    4: '#ef6b4e',  # Orange
    5: '#ef4444'   # Red - hard
}


def get_fdr_color(fdr: int) -> str:
    """Get color for fixture difficulty rating."""
    return FDR_COLORS.get(fdr, '#888')


def round_df(df: pd.DataFrame, max_dp: int = 2) -> pd.DataFrame:
    """Round all numeric float columns and format as strings for clean display.
    
    Converts floats to formatted strings so Streamlit's Arrow renderer
    cannot add extra trailing decimals.
    Columns named Price/Cost keep 1 dp; all other floats use max_dp.
    """
    df = df.copy()
    _price_names = {'Price', 'Cost', 'price', 'cost', 'now_cost'}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            dp = 1 if col in _price_names else max_dp
            df[col] = df[col].round(dp).map(
                lambda x, d=dp: f'{x:.{d}f}' if pd.notna(x) else ''
            )
    return df


def style_df_with_injuries(df: pd.DataFrame, players_df: pd.DataFrame = None, player_col: str = 'Player') -> pd.DataFrame:
    """
    Apply injury-based row coloring to a dataframe.
    Returns a styled dataframe if injury_highlight is enabled.
    
    Args:
        df: DataFrame to style
        players_df: Full players DataFrame with injury data (optional, uses session_state if not provided)
        player_col: Name of the column containing player names (web_name)
    """
    import streamlit as st
    
    # Check if highlighting is disabled
    if not st.session_state.get('injury_highlight', True):
        return round_df(df)
    
    # Get players_df from session state if not provided
    if players_df is None:
        players_df = st.session_state.get('players_df')
    
    if players_df is None or player_col not in df.columns:
        return round_df(df)
    
    # Round all floats to 3dp first
    df = round_df(df)
    
    # Build lookup of player name -> injury chance
    injury_lookup = {}
    for _, p in players_df.iterrows():
        name = p.get('web_name', '')
        status = p.get('status', 'a')
        chance = _safe_int(p.get('chance_of_playing_next_round'), 100)
        
        # Determine effective chance based on status
        if status in ['i', 's', 'u']:
            chance = 0
        elif status == 'd' and chance == 100:
            chance = 50
        
        injury_lookup[name] = chance
    
    # Get chance values for each row in display df
    chance_values = [injury_lookup.get(name, 100) for name in df[player_col]]
    
    def style_rows(display_df):
        styles = []
        for i in range(len(display_df)):
            chance = chance_values[i] if i < len(chance_values) else 100
            if chance == 0:
                bg = 'background-color: rgba(239, 68, 68, 0.4)'  # Red - out
            elif chance <= 25:
                bg = 'background-color: rgba(239, 68, 68, 0.3)'  # Dark red
            elif chance <= 50:
                bg = 'background-color: rgba(249, 115, 22, 0.3)'  # Orange
            elif chance <= 75:
                bg = 'background-color: rgba(245, 158, 11, 0.25)'  # Amber
            elif chance < 100:
                bg = 'background-color: rgba(250, 204, 21, 0.15)'  # Light yellow
            else:
                bg = ''  # No styling for healthy players
            styles.append([bg] * len(display_df.columns))
        return pd.DataFrame(styles, index=display_df.index, columns=display_df.columns)
    
    return df.style.apply(style_rows, axis=None)


# ── Rotation Risk ──

def get_rotation_risk(player_row) -> Dict:
    """
    Assess rotation risk based on minutes played relative to available minutes.
    Returns dict with risk level, color, minutes_pct.
    """
    minutes = safe_numeric(pd.Series([player_row.get('minutes', 0)])).iloc[0]
    starts = safe_numeric(pd.Series([player_row.get('starts', 0)])).iloc[0]
    # Total possible minutes: 90 * games played by team (estimated from events_played/appearances)
    total_points = safe_numeric(pd.Series([player_row.get('total_points', 0)])).iloc[0]

    # Use starts as a proxy; assume current GW count from minutes / 90 as max
    # Better proxy: if the player has data, estimate from their starts vs total GWs completed
    status = player_row.get('status', 'a')
    chance = _safe_int(player_row.get('chance_of_playing_next_round'), 100)

    if status in UNAVAILABLE_STATUSES or chance == 0:
        return {'risk': 'Unavailable', 'color': '#ef4444', 'icon': 'X', 'minutes_pct': 0.0}

    # minutes per start
    if starts > 0:
        mpg = minutes / starts
    else:
        mpg = 0

    # Estimate total team games from event history (rough: use total_points > 0 appearances)
    appearances = safe_numeric(pd.Series([player_row.get('starts', 0)])).iloc[0]
    # More reliable: use 'minutes' relative to possible minutes
    # Rough estimate: number of GWs played so far based on events
    # Without exact GW count, use starts ratio vs team average
    # Simpler: classify based on minutes_per_game
    if mpg >= 80:
        pct = ROTATION_SAFE_MINUTES_PCT + 0.1
    elif mpg >= 65:
        pct = ROTATION_SAFE_MINUTES_PCT
    elif mpg >= 45:
        pct = ROTATION_MODERATE_PCT
    else:
        pct = 0.4

    if chance < 100 and chance > 0:
        # Adjust for doubt
        pct = min(pct, chance / 100.0)

    if pct >= ROTATION_SAFE_MINUTES_PCT:
        return {'risk': 'Low', 'color': '#22c55e', 'icon': '', 'minutes_pct': pct}
    elif pct >= ROTATION_MODERATE_PCT:
        return {'risk': 'Moderate', 'color': '#f59e0b', 'icon': '~', 'minutes_pct': pct}
    else:
        return {'risk': 'High', 'color': '#ef4444', 'icon': '!', 'minutes_pct': pct}


# ── Ownership Tiers ──

def get_ownership_tier(ownership_pct: float) -> Dict:
    """
    Classify player by ownership bracket.
    Returns dict with tier name, color, description.
    """
    if ownership_pct >= OWNERSHIP_TEMPLATE_THRESHOLD:
        return {'tier': 'Template', 'color': '#3b82f6', 'desc': 'Core pick, most managers own'}
    elif ownership_pct >= OWNERSHIP_POPULAR_THRESHOLD:
        return {'tier': 'Popular', 'color': '#22c55e', 'desc': 'Well-owned, mainstream pick'}
    elif ownership_pct >= OWNERSHIP_DIFFERENTIAL_THRESHOLD:
        return {'tier': 'Enabler', 'color': '#f59e0b', 'desc': 'Moderate ownership, some edge'}
    else:
        return {'tier': 'Differential', 'color': '#a855f7', 'desc': 'Low owned, high ceiling/risk'}


def classify_ownership_column(df: pd.DataFrame, col='selected_by_percent') -> pd.Series:
    """Add ownership tier column to a DataFrame."""
    own = safe_numeric(df.get(col, pd.Series([0] * len(df))))
    return own.apply(lambda x: get_ownership_tier(x)['tier'])


# ── Availability Summary ──

def get_availability_badge(player_row) -> str:
    """
    Return a short text badge combining injury status + rotation risk.
    For use in table columns.
    """
    injury = get_injury_status(player_row)
    if injury['status'] in ('injured', 'suspended', 'unavailable'):
        return f"[{injury['icon']}] {injury['status'].title()}"
    if injury['status'] == 'doubtful':
        return f"[?] {injury['chance']}%"

    rotation = get_rotation_risk(player_row)
    if rotation['risk'] == 'High':
        return "[!] Rotation"
    elif rotation['risk'] == 'Moderate':
        return "[~] Moderate"
    return ""


def add_availability_columns(df: pd.DataFrame, players_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add 'Availability' and 'Ownership Tier' columns to a display DataFrame.
    Matches on 'web_name' or 'Player' column.
    """
    import streamlit as st

    if players_df is None:
        players_df = st.session_state.get('players_df')
    if players_df is None:
        return df

    df = df.copy()
    # Find the player name column
    name_col = None
    for candidate in ['web_name', 'Player', 'player', 'Name']:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        return df

    # Build lookup
    avail_lookup = {}
    tier_lookup = {}
    for _, p in players_df.iterrows():
        name = p.get('web_name', '')
        avail_lookup[name] = get_availability_badge(p)
        own = safe_numeric(pd.Series([p.get('selected_by_percent', 0)])).iloc[0]
        tier_lookup[name] = get_ownership_tier(own)['tier']

    df['Avail'] = df[name_col].map(avail_lookup).fillna('')
    df['Tier'] = df[name_col].map(tier_lookup).fillna('')

    return df

