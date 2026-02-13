"""Understat data integration for FPL Strategy Engine.

Fetches xG, xA, npxG, xGChain, xGBuildup from understat.com
and merges into the FPL player DataFrame to enhance EP calculation.

Uses requests + regex to extract JSON from Understat pages --
no async or extra dependencies required.
"""

import json
import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Season auto-detection ──
# Understat uses the starting year: 2025/26 season -> 2025
_now = datetime.now()
UNDERSTAT_SEASON = _now.year if _now.month >= 8 else _now.year - 1

# ── FPL team_name -> Understat team_title ──
# Covers all current and recently promoted sides; unknown names fall through as-is.
TEAM_NAME_MAP: Dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
}

# ── Per-position FPL points per goal/assist ──
GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_PTS = 3
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}

# EP adjustment caps: max fraction of base EP the Understat adjustment can shift
EP_ADJUST_CAP = 0.20

# Position-specific (goal_weight, assist_weight, buildup_weight)
POS_WEIGHTS = {
    "FWD": (0.60, 0.25, 0.05),
    "MID": (0.35, 0.40, 0.15),
    "DEF": (0.10, 0.15, 0.25),
    "GKP": (0.00, 0.00, 0.05),
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_understat_league_players(season: int = UNDERSTAT_SEASON) -> List[Dict]:
    """Fetch all EPL player data from Understat for *season*."""
    url = f"https://understat.com/league/EPL/{season}"
    try:
        resp = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 FPL-Strategy-Engine"
        })
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Understat fetch failed: %s", exc)
        return []

    match = re.search(
        r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)",
        resp.text,
    )
    if not match:
        logger.warning("Could not locate playersData in Understat page")
        return []

    try:
        raw = match.group(1)
        decoded = raw.encode("utf-8").decode("unicode_escape")
        return json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Understat JSON parse error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Strip accents, lowercase, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_name.lower().split())


def _last(name: str) -> str:
    parts = name.strip().split()
    return _norm(parts[-1]) if parts else ""


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def _build_understat_df(raw: List[Dict]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)

    num_cols = [
        "games", "time", "goals", "xG", "assists", "xA",
        "shots", "key_passes", "npg", "npxG", "xGChain", "xGBuildup",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["minutes_90"] = df["time"] / 90.0
    has_mins = df["minutes_90"] > 1.0  # at least ~90 total minutes

    per90_sources = ["xG", "xA", "npxG", "xGChain", "xGBuildup", "shots", "key_passes"]
    for m in per90_sources:
        col = f"{m}_per90"
        df[col] = 0.0
        df.loc[has_mins, col] = df.loc[has_mins, m] / df.loc[has_mins, "minutes_90"]

    # Over/under-performance vs model
    df["goal_overperf"] = df["goals"] - df["xG"]
    df["assist_overperf"] = df["assists"] - df["xA"]

    # Normalised identifiers for matching
    df["_name_norm"] = df["player_name"].apply(_norm)
    df["_last_name"] = df["player_name"].apply(_last)

    return df


# ---------------------------------------------------------------------------
# Matching Understat -> FPL
# ---------------------------------------------------------------------------

def match_understat_to_fpl(
    fpl_df: pd.DataFrame,
    ustat_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match Understat rows to FPL rows (team + name) and merge columns."""
    if ustat_df.empty or fpl_df.empty:
        return fpl_df

    result = fpl_df.copy()

    # Normalised FPL names
    if "second_name" in result.columns:
        result["_fpl_full"] = (
            result.get("first_name", pd.Series([""] * len(result))).fillna("")
            + " "
            + result["second_name"].fillna("")
        ).apply(_norm)
        result["_fpl_last"] = result["second_name"].fillna("").apply(_last)
    else:
        result["_fpl_full"] = result["web_name"].fillna("").apply(_norm)
        result["_fpl_last"] = result["web_name"].fillna("").apply(_last)
    result["_fpl_web"] = result["web_name"].fillna("").apply(_norm)

    # Team mapping
    fpl_to_ustat = {}
    if "team_name" in result.columns:
        for tn in result["team_name"].dropna().unique():
            fpl_to_ustat[tn] = TEAM_NAME_MAP.get(tn, tn)

    # Build lookup dicts: (ustat_team, normalised_name) -> row
    by_full: Dict[Tuple[str, str], pd.Series] = {}
    by_last: Dict[Tuple[str, str], Optional[pd.Series]] = {}
    for _, urow in ustat_df.iterrows():
        team = urow.get("team_title", "")
        by_full[(team, urow["_name_norm"])] = urow
        key = (team, urow["_last_name"])
        by_last[key] = None if key in by_last else urow  # None = ambiguous

    # Columns to write
    merge_map = {
        "us_xG": "xG", "us_xA": "xA", "us_npxG": "npxG",
        "us_xGChain": "xGChain", "us_xGBuildup": "xGBuildup",
        "us_xG_per90": "xG_per90", "us_xA_per90": "xA_per90",
        "us_npxG_per90": "npxG_per90",
        "us_xGChain_per90": "xGChain_per90",
        "us_xGBuildup_per90": "xGBuildup_per90",
        "us_shots_per90": "shots_per90",
        "us_key_passes_per90": "key_passes_per90",
        "us_goal_overperf": "goal_overperf",
        "us_assist_overperf": "assist_overperf",
    }
    for col in merge_map:
        result[col] = 0.0

    matched = 0
    for idx, row in result.iterrows():
        ustat_team = fpl_to_ustat.get(row.get("team_name", ""), "")

        # Priority: full name, then last name, then web_name
        urow = by_full.get((ustat_team, row["_fpl_full"]))
        if urow is None:
            candidate = by_last.get((ustat_team, row["_fpl_last"]))
            if candidate is not None:
                urow = candidate
        if urow is None:
            urow = by_full.get((ustat_team, row["_fpl_web"]))
        if urow is None:
            web_last = _last(row["_fpl_web"])
            candidate = by_last.get((ustat_team, web_last))
            if candidate is not None:
                urow = candidate

        if urow is not None:
            for dest, src in merge_map.items():
                result.at[idx, dest] = urow.get(src, 0.0)
            matched += 1

    result.drop(columns=["_fpl_full", "_fpl_last", "_fpl_web"], inplace=True, errors="ignore")
    logger.info("Understat matched %d / %d FPL players", matched, len(result))
    return result


# ---------------------------------------------------------------------------
# EP adjustment
# ---------------------------------------------------------------------------

def adjust_ep_with_understat(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust expected_points using Understat xG/xA regression signals.

    - Goals > xG (overperforming) -> slight EP decrease (regression risk)
    - Goals < xG (underperforming) -> slight EP increase (bounce-back)
    - Same for assists vs xA
    - xGBuildup per 90 adds a small creativity/involvement bonus
    - Adjustment capped at +/-20 %% of base EP
    """
    if "us_xG" not in df.columns or "expected_points" not in df.columns:
        return df

    result = df.copy()
    has_data = result["us_xG"] > 0
    result["_ep_adj"] = 0.0

    for pos, (gw, aw, bw) in POS_WEIGHTS.items():
        mask = has_data & (result["position"] == pos)
        if not mask.any():
            continue

        # Negative overperf -> underperforming -> positive adjustment (bounce back)
        goal_adj = -result.loc[mask, "us_goal_overperf"] * gw
        assist_adj = -result.loc[mask, "us_assist_overperf"] * aw
        buildup = result.loc[mask, "us_xGBuildup_per90"] * bw * 10

        raw = goal_adj + assist_adj + buildup
        cap = result.loc[mask, "expected_points"].clip(lower=0.1) * EP_ADJUST_CAP
        result.loc[mask, "_ep_adj"] = raw.clip(lower=-cap, upper=cap)

    result["expected_points"] = (result["expected_points"] + result["_ep_adj"]).clip(lower=0)
    result.drop(columns=["_ep_adj"], inplace=True)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_with_understat(fpl_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch Understat data, match to FPL players, adjust EP.

    Caches the raw Understat response in ``st.session_state`` so repeated
    calls within a session do not trigger additional HTTP requests.
    """
    import streamlit as st

    cache_key = "_understat_raw"
    if not force_refresh and cache_key in st.session_state:
        raw = st.session_state[cache_key]
    else:
        raw = fetch_understat_league_players()
        st.session_state[cache_key] = raw

    if not raw:
        return fpl_df

    ustat_df = _build_understat_df(raw)
    merged = match_understat_to_fpl(fpl_df, ustat_df)
    adjusted = adjust_ep_with_understat(merged)
    return adjusted
