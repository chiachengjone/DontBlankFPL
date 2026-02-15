"""Helper utility functions for FPL Strategy Engine."""

import pandas as pd
import numpy as np
import unicodedata
from typing import Dict, List, Optional

from config import (
    OWNERSHIP_TEMPLATE_THRESHOLD,
    OWNERSHIP_POPULAR_THRESHOLD,
    OWNERSHIP_DIFFERENTIAL_THRESHOLD,
    ROTATION_SAFE_MINUTES_PCT,
    ROTATION_MODERATE_PCT,
    UNAVAILABLE_STATUSES,
)


def normalize_name(name: str) -> str:
    """
    Normalize player name by removing accents and special characters.
    E.g., 'João Pedro' → 'Joao Pedro', 'Estêvão' → 'Estevao'
    """
    if not name:
        return ''
    # Normalize to NFD (decomposed form) then filter out combining marks
    normalized = unicodedata.normalize('NFD', name)
    # Remove combining diacritical marks (accents)
    ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return ascii_name


def search_players(df: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    """
    Search players by name with 'starts with' matching on first or last name.
    Returns matching players with full_name.
    """
    if not query or len(query) < 1:
        return pd.DataFrame()
    
    query_normalized = normalize_name(query.lower().strip())
    
    # Create normalized components if not exists
    if 'first_normalized' not in df.columns:
        df = df.copy()
        df['first_normalized'] = df['first_name'].apply(lambda x: normalize_name(str(x).lower()))
        df['second_normalized'] = df['second_name'].apply(lambda x: normalize_name(str(x).lower()))
    
    # Match if query starts with the beginning of either first or second name
    mask = (
        df['first_normalized'].str.startswith(query_normalized, na=False) |
        df['second_normalized'].str.startswith(query_normalized, na=False)
    )
    matches = df[mask].copy()
    
    if not matches.empty:
        # Sort by ownership (most popular first) for autocomplete
        if 'selected_by_percent' in matches.columns:
            matches = matches.sort_values('selected_by_percent', ascending=False)
            
    return matches.head(limit)


def safe_numeric(series, default=0):
    """Safely convert series or scalar to numeric, handling NaN values."""
    try:
        if isinstance(series, (pd.Series, pd.Index, np.ndarray)):
            return pd.to_numeric(series, errors='coerce').fillna(default)
        # Scalar case
        val = pd.to_numeric(series, errors='coerce')
        return val if pd.notnull(val) else default
    except:
        return default


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


def style_df_with_injuries(df: pd.DataFrame, players_df: pd.DataFrame = None, player_col: str = 'Player', format_dict: dict = None) -> pd.DataFrame:
    """
    Apply injury-based row coloring to a dataframe.
    Returns a styled dataframe if injury_highlight is enabled.
    
    Args:
        df: DataFrame to style
        players_df: Full players DataFrame with injury data (optional, uses session_state if not provided)
        player_col: Name of the column containing player names (web_name)
        format_dict: Optional dict of column format strings, e.g. {'Price': '£{:.1f}m'}
    """
    import streamlit as st
    
    def _apply_safe_format(styled_or_df, fmt_dict):
        """Apply format_dict safely, handling non-numeric values."""
        if not fmt_dict:
            return styled_or_df
        # If it's a plain DataFrame, convert to Styler first
        if isinstance(styled_or_df, pd.DataFrame):
            styled_or_df = styled_or_df.style
        safe_fmt = {}
        for col, fmt in fmt_dict.items():
            if col in df.columns:
                def _make_safe(f):
                    def _fmt(v):
                        try:
                            return f.format(v)
                        except (ValueError, TypeError):
                            return str(v) if pd.notna(v) else ''
                    return _fmt
                safe_fmt[col] = _make_safe(fmt)
        return styled_or_df.format(safe_fmt, na_rep='')
    
    # Check if highlighting is disabled
    if not st.session_state.get('injury_highlight', True):
        result = round_df(df)
        return _apply_safe_format(result, format_dict) if format_dict else result
    
    # Get players_df from session state if not provided
    if players_df is None:
        players_df = st.session_state.get('players_df')
    
    if players_df is None or player_col not in df.columns:
        result = round_df(df)
        return _apply_safe_format(result, format_dict) if format_dict else result
    
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
    
    styled = df.style.apply(style_rows, axis=None)
    return _apply_safe_format(styled, format_dict) if format_dict else styled


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


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED POINTS (xP) COLUMN SEMANTICS - EXPLICIT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# These column names have SPECIFIC meanings. Do NOT use them interchangeably:
#
# ┌─────────────────────────┬─────────────────────────────────────────────────────┐
# │ Column Name             │ Description                                          │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ consensus_ep            │ MODEL xP - Weighted blend (ML 40%, Poisson 40%,     │
# │                         │ FPL 20%) based on active_models. THE PRIMARY xP.    │
# │                         │ Use this for all UI displays labeled "Model xP".    │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ avg_consensus_ep        │ MODEL xP / horizon - Per-GW average when horizon>1. │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ expected_points_poisson │ POISSON xP - Single-model output from poisson_ep.py.│
# │                         │ Statistical model using xG/xA and fixture data.     │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ ml_pred                 │ ML xP - XGBoost ensemble prediction.               │
# │                         │ Single-GW prediction, scale by horizon if needed.   │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ ep_next / ep_next_num   │ FPL xP - FPL's official expected points estimate.   │
# │                         │ Raw from API, single-GW only.                       │
# ├─────────────────────────┼─────────────────────────────────────────────────────┤
# │ expected_points         │ DEPRECATED - Legacy fallback column. Maps to        │
# │                         │ expected_points_poisson in most contexts. Avoid.    │
# └─────────────────────────┴─────────────────────────────────────────────────────┘
#
# USAGE RULES:
# 1. For UI displays labeled "Model xP" → use consensus_ep
# 2. For Poisson-specific charts → use expected_points_poisson  
# 3. For ML-specific displays → use ml_pred
# 4. For FPL official EP → use ep_next_num
# 5. NEVER use expected_points or ep_next when consensus_ep is available
#
# ══════════════════════════════════════════════════════════════════════════════


def calculate_consensus_ep(df: pd.DataFrame, active_models: List[str], horizon: int = 1) -> pd.DataFrame:
    """
    Calculate weighted Consensus EP based on globally selected models.
    
    Weights:
    - ML: 40%
    - Poisson: 40%
    - FPL: 20%
    
    Weights are normalized based on which models are enabled.
    
    This function auto-loads ML predictions from st.session_state['ml_predictions']
    if ml_pred is not already present in the DataFrame. This ensures consistent
    Model xP values across ALL tabs without requiring each tab to manually load ML.
    """
    df = df.copy()
    
    # Model keys mapping (consistency with app.py/tabs)
    model_col_map = {
        'ml': 'ml_pred',
        'poisson': 'poisson_ep',
        'fpl': 'ep_next_num'
    }
    
    # Base weights
    base_weights = {
        'ml': 0.4,
        'poisson': 0.4,
        'fpl': 0.2
    }
    
    # Filter for active models and get weights
    # Normalize keys to lowercase for safety
    active_keys = [m.lower() for m in active_models]
    active_weights = {m: base_weights[m] for m in active_keys if m in base_weights}
    
    # Normalize weights
    total_weight = sum(active_weights.values())
    if total_weight == 0:
        # Fallback to all if nothing valid selected
        active_weights = base_weights
        total_weight = sum(base_weights.values())
    
    normalized_weights = {m: w/total_weight for m, w in active_weights.items()}
    
    # ── Ensure baseline columns exist ──
    
    # Poisson EP: from calculate_poisson_ep_for_dataframe output
    if 'poisson_ep' not in df.columns:
        if 'expected_points_poisson' in df.columns:
            df['poisson_ep'] = safe_numeric(df['expected_points_poisson'])
        else:
            df['poisson_ep'] = safe_numeric(df.get('ep_next_num', df.get('ep_next', 0)))
            
    if 'ep_next_num' not in df.columns:
        df['ep_next_num'] = safe_numeric(df.get('ep_next', 0))
    
    # ML Predictions: auto-load from session state if not already in DataFrame
    # This is the SINGLE source of truth for ML predictions, ensuring all tabs
    # get the same values without duplicating ML-loading boilerplate.
    if 'ml_pred' not in df.columns:
        _ml_loaded = False
        try:
            import streamlit as _st
            if 'ml_predictions' in _st.session_state and 'id' in df.columns:
                ml_preds = _st.session_state['ml_predictions']
                if ml_preds:
                    df['ml_pred'] = df['id'].apply(
                        lambda pid: ml_preds[pid].predicted_points if pid in ml_preds else 0.0
                    ).round(2)
                    _ml_loaded = True
        except Exception:
            pass
        
        if not _ml_loaded:
            if 'ml_ep' in df.columns:
                df['ml_pred'] = safe_numeric(df['ml_ep'])
            else:
                # True fallback: no ML data available at all
                df['ml_pred'] = df['poisson_ep'] * 0.9
    
    # Calculation
    consensus = pd.Series(0.0, index=df.index)
    for model, weight in normalized_weights.items():
        col = model_col_map[model]
        if col in df.columns:
            val = safe_numeric(df[col])
            
            # Smart Horizon Scaling
            # If horizon > 1, we expect to return a Total sum for the window
            if horizon > 1:
                if model == 'fpl':
                    # FPL ep_next is always single-GW raw, so scale it
                    val = val * horizon
                elif model == 'ml':
                    # ML predictions are usually single-GW in players_df, so scale
                    # Only scale if typical single-GW range (e.g. max < 15 for 1 GW)
                    if val.max() < 15: 
                        val = val * horizon
                # Poisson is assumed already totaled (sum of GWs) by engine loop
            
            consensus += val * weight
            
    df['consensus_ep'] = consensus.round(2)
    
    # Calculate average if horizon > 1
    if horizon > 1:
        df['avg_consensus_ep'] = (df['consensus_ep'] / horizon).round(2)
    else:
        df['avg_consensus_ep'] = df['consensus_ep']
        
    return df


def calculate_enhanced_captain_score(row: pd.Series, active_models: List[str]) -> float:
    """
    Calculate a sophisticated, position-aware captain score (0-10+).
    Standardized for use in Captain tab and Dashboard.
    
    Weights: 60% Core Models, 30% Position Bonus, 10% Meta/Safety.
    """
    # 1. Core Models (60% weight)
    model_count = len(active_models)
    
    # Map model keys to data columns
    ml_val = safe_numeric(row.get('ml_pred', row.get('ml_ep', 0)))
    poisson_val = safe_numeric(row.get('poisson_ep', row.get('expected_points_poisson', 0)))
    fpl_val = safe_numeric(row.get('ep_next_num', row.get('ep_next', 0)))
    
    if model_count == 3:
        # Global: ML 40%, Poisson 40%, FPL 20% -> scaled to 60% total
        core = (ml_val * 0.24 + poisson_val * 0.24 + fpl_val * 0.12)
    elif model_count == 2:
        m_set = set([m.lower() for m in active_models])
        if 'ml' in m_set and 'poisson' in m_set:
            core = (ml_val * 0.30 + poisson_val * 0.30)
        elif 'ml' in m_set and 'fpl' in m_set:
            core = (ml_val * 0.40 + fpl_val * 0.20)
        elif 'poisson' in m_set and 'fpl' in m_set:
            core = (poisson_val * 0.40 + fpl_val * 0.20)
        else:
            core = (ml_val * 0.30 + poisson_val * 0.30) # Default
    else:
        # Only 1 model active
        active = active_models[0].lower() if active_models else 'fpl'
        val = ml_val if active == 'ml' else poisson_val if active == 'poisson' else fpl_val
        core = val * 0.6
    
    # 2. Position-Specific Potential (30% weight)
    if row.get('position') in ['GKP', 'DEF']:
        p_cs = safe_numeric(row.get('poisson_p_cs', 0.2))
        cbit_p = safe_numeric(row.get('cbit_prob', 0.3))
        pos_bonus = (p_cs * 0.15 + cbit_p * 0.15) * 10
    else:
        threat = safe_numeric(row.get('threat_momentum', 0.5))
        matchup = safe_numeric(row.get('matchup_quality', 1.0))
        # Scaled to ~0-3 range
        pos_bonus = float(np.clip(threat * 0.15 * 10 + (matchup / 2) * 0.15 * 10, 0, 3))
        
    # 3. Meta & Safety (10% weight)
    # selected_by_percent can be a string or float from FPL API
    eo = safe_numeric(row.get('selected_by_percent', 0)) / 100
    form_norm = safe_numeric(row.get('form', 0)) / 15
    meta = (eo * 0.05 + form_norm * 0.05) * 10
    
    # 4. Injury Risk Adjustment
    chance = safe_numeric(row.get('chance_of_playing_next_round', 100)) / 100
    
    return float((core + pos_bonus + meta) * chance)


def get_consensus_label(active_models: List[str], horizon: int = 1) -> str:
    """Return dynamic label for Model xP based on number of active models."""
    if not active_models:
        return "xP"
        
    if len(active_models) == 1:
        model = active_models[0].lower()
        label_map = {
            'ml': 'ML xP',
            'poisson': 'Poisson xP',
            'fpl': 'FPL xP'
        }
        name = label_map.get(model, 'xP')
        return f"{name} x{horizon}" if horizon > 1 else name
    
    return f"Model xP ({horizon}GW)" if horizon > 1 else "Model xP"
