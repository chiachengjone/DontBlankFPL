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

def fetch_understat_league_data(season: int = UNDERSTAT_SEASON) -> Tuple[List[Dict], Dict]:
    """Fetch both player-level and team-level data from Understat in one request.

    Uses the ``POST /getLeagueData/EPL/<season>`` AJAX endpoint which returns
    JSON with ``players``, ``teams``, and ``dates`` keys.  Falls back to the
    legacy HTML-scraping approach if the AJAX call fails.

    Returns:
        Tuple of (players_raw_list, teams_raw_dict)
    """

    # ── Primary: AJAX JSON endpoint (fast, reliable) ──
    ajax_url = f"https://understat.com/getLeagueData/EPL/{season}"
    try:
        resp = requests.post(
            ajax_url,
            data={},
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://understat.com/league/EPL/{season}",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        players_raw = data.get("players", [])
        teams_raw = data.get("teams", {})
        if players_raw:
            logger.info(
                "Understat AJAX: %d players, %d teams fetched",
                len(players_raw), len(teams_raw),
            )
            return players_raw, teams_raw
        logger.warning("Understat AJAX returned empty players list")
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Understat AJAX fetch failed: %s — trying HTML fallback", exc)

    # ── Fallback: HTML scraping (legacy, kept for resilience) ──
    html_url = f"https://understat.com/league/EPL/{season}"
    try:
        resp = requests.get(html_url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 FPL-Strategy-Engine"
        })
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Understat HTML fetch also failed: %s", exc)
        return [], {}

    html = resp.text

    # Extract playersData
    players_raw: List[Dict] = []
    pmatch = re.search(
        r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", html
    )
    if pmatch:
        try:
            decoded = pmatch.group(1).encode("utf-8").decode("unicode_escape")
            players_raw = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Understat players JSON parse error: %s", exc)
    else:
        logger.warning("Could not locate playersData in Understat page")

    # Extract teamsData
    teams_raw: Dict = {}
    tmatch = re.search(
        r"var\s+teamsData\s*=\s*JSON\.parse\('(.+?)'\)", html
    )
    if tmatch:
        try:
            decoded = tmatch.group(1).encode("utf-8").decode("unicode_escape")
            teams_raw = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Understat teams JSON parse error: %s", exc)

    return players_raw, teams_raw


def fetch_understat_league_players(season: int = UNDERSTAT_SEASON) -> List[Dict]:
    """Fetch all EPL player data from Understat for *season*."""
    players, _ = fetch_understat_league_data(season)
    return players


def build_team_stats_df(teams_raw: Dict) -> pd.DataFrame:
    """Build team-level xG For / xGA DataFrame from Understat *teamsData*.

    Each team entry has a ``history`` list of per-match records with
    ``xG`` (created) and ``xGA`` (conceded).  We aggregate to per-match
    averages so the Poisson engine can use **real opponent xGA** instead
    of the rudimentary FDR proxy.

    Returns:
        DataFrame with columns:
            understat_team, games_played, xG_for_per90, xGA_per90, etc.
    """
    if not teams_raw:
        return pd.DataFrame()

    rows = []
    for team_id, team_data in teams_raw.items():
        title = team_data.get("title", "")
        history = team_data.get("history", [])
        if not history:
            continue

        games = len(history)
        total_xg = sum(float(m.get("xG", 0)) for m in history)
        total_xga = sum(float(m.get("xGA", 0)) for m in history)
        total_npxg = sum(float(m.get("npxG", 0)) for m in history)
        total_npxga = sum(float(m.get("npxGA", 0)) for m in history)
        total_scored = sum(int(m.get("scored", 0)) for m in history)
        total_missed = sum(int(m.get("missed", 0)) for m in history)

        rows.append({
            "understat_team": title,
            "understat_id": int(team_id),
            "games_played": games,
            "xG_for_total": total_xg,
            "xGA_total": total_xga,
            "xG_for_per90": total_xg / games,
            "xGA_per90": total_xga / games,
            "npxG_for_per90": total_npxg / games,
            "npxGA_per90": total_npxga / games,
            "goals_per90": total_scored / games,
            "conceded_per90": total_missed / games,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Built team stats for %d teams (avg xG=%.2f, avg xGA=%.2f)",
        len(df),
        df["xG_for_per90"].mean() if not df.empty else 0,
        df["xGA_per90"].mean() if not df.empty else 0,
    )
    return df


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


def _levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _fuzzy_match_score(name1: str, name2: str) -> float:
    """Return similarity score (0-1) using Levenshtein distance."""
    n1, n2 = _norm(name1), _norm(name2)
    if not n1 or not n2:
        return 0.0
    max_len = max(len(n1), len(n2))
    distance = _levenshtein(n1, n2)
    return 1.0 - (distance / max_len)


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
        "us_games": "games", "us_goals": "goals", "us_assists": "assists",
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
    fuzzy_matched = 0
    FUZZY_THRESHOLD = 0.75  # Minimum similarity for fuzzy match
    
    # Build list of all Understat players per team for fuzzy fallback
    ustat_by_team: Dict[str, List[pd.Series]] = {}
    for _, urow in ustat_df.iterrows():
        team = urow.get("team_title", "")
        if team not in ustat_by_team:
            ustat_by_team[team] = []
        ustat_by_team[team].append(urow)
    
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
        
        # Fuzzy matching fallback (Levenshtein distance)
        if urow is None and ustat_team in ustat_by_team:
            best_score = 0.0
            best_match = None
            fpl_name = row["_fpl_full"]
            for candidate_row in ustat_by_team[ustat_team]:
                score = _fuzzy_match_score(fpl_name, candidate_row["_name_norm"])
                if score > best_score:
                    best_score = score
                    best_match = candidate_row
            if best_score >= FUZZY_THRESHOLD and best_match is not None:
                urow = best_match
                fuzzy_matched += 1

        if urow is not None:
            for dest, src in merge_map.items():
                result.at[idx, dest] = urow.get(src, 0.0)
            matched += 1

    result.drop(columns=["_fpl_full", "_fpl_last", "_fpl_web"], inplace=True, errors="ignore")
    logger.info("Understat matched %d / %d FPL players (%d via fuzzy)", matched, len(result), fuzzy_matched)
    return result


# ---------------------------------------------------------------------------
# EP adjustment
# ---------------------------------------------------------------------------

def calculate_advanced_ep(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Expected Points using the Advanced Additive Model (2025/26 rules).

    Components:
        EP_app  – Appearance points (P60+)
        EP_att  – Attacking points (xG/xA with Poisson adjustment)
        EP_def  – Defensive points (CS + DefCon + xGC penalty)
        EP_bonus – Bonus points expectation (~0.10 * xGI)

    Uses Understat xG/xA when available; falls back to FPL expected_goals/expected_assists.
    Applies npxG adjustment for non-penalty takers.
    """
    import math

    if "expected_points" not in df.columns:
        return df

    result = df.copy()

    # ── Position-specific scoring ──
    GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
    ASSIST_PTS = 3
    CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
    DEFCON_THRESHOLD = {"GKP": 12, "DEF": 10, "MID": 12, "FWD": 12}
    DEFCON_BONUS = 2  # Capped at 2 pts per match

    # ── Helper: Poisson P(k>=1) for xG ──
    def p_score_at_least_one(xg: float) -> float:
        if xg <= 0:
            return 0.0
        return 1.0 - math.exp(-xg)

    # ── Determine data sources ──
    # IMPORTANT: Use PER-90 stats for single-GW EP, not season totals
    has_ustat = "us_xG_per90" in result.columns and result["us_xG_per90"].sum() > 0
    xg_col = "us_xG_per90" if has_ustat else "expected_goals_per_90"
    xa_col = "us_xA_per90" if has_ustat else "expected_assists_per_90"
    npxg_col = "us_npxG_per90" if has_ustat else None  # FPL doesn't have npxG

    for c in (xg_col, xa_col):
        if c not in result.columns:
            # Fall back to season totals divided by games
            if c == "us_xG_per90" and "us_xG" in result.columns:
                games = pd.to_numeric(result.get("us_games", 1), errors="coerce").fillna(1).clip(lower=1)
                result[c] = pd.to_numeric(result["us_xG"], errors="coerce").fillna(0) / games
            elif c == "us_xA_per90" and "us_xA" in result.columns:
                games = pd.to_numeric(result.get("us_games", 1), errors="coerce").fillna(1).clip(lower=1)
                result[c] = pd.to_numeric(result["us_xA"], errors="coerce").fillna(0) / games
            elif "expected_goals" in result.columns and "per_90" not in c:
                # FPL fallback: season totals / games
                minutes = pd.to_numeric(result.get("minutes", 90), errors="coerce").fillna(90)
                games = (minutes / 90).clip(lower=1)
                result["expected_goals_per_90"] = pd.to_numeric(result.get("expected_goals", 0), errors="coerce").fillna(0) / games
                result["expected_assists_per_90"] = pd.to_numeric(result.get("expected_assists", 0), errors="coerce").fillna(0) / games
            else:
                result[c] = 0.0
        result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)

    if npxg_col and npxg_col in result.columns:
        result[npxg_col] = pd.to_numeric(result[npxg_col], errors="coerce").fillna(0.0)
    elif npxg_col and "us_npxG" in result.columns:
        games = pd.to_numeric(result.get("us_games", 1), errors="coerce").fillna(1).clip(lower=1)
        result[npxg_col] = pd.to_numeric(result["us_npxG"], errors="coerce").fillna(0) / games

    # ── EP_app: Appearance probability ──
    # Use minutes / (games_played * 90) as proxy for P(60+)
    if "minutes" in result.columns:
        result["minutes"] = pd.to_numeric(result["minutes"], errors="coerce").fillna(0)
        # Approx games from Understat or FPL
        games_col = "us_games" if "us_games" in result.columns else None
        if games_col is None:
            # Estimate from minutes
            result["_games_est"] = (result["minutes"] / 90).clip(lower=0.1)
        else:
            result["_games_est"] = pd.to_numeric(result[games_col], errors="coerce").fillna(1).clip(lower=0.1)
        result["_p60"] = (result["minutes"] / (result["_games_est"] * 90)).clip(upper=1.0)
    else:
        result["_p60"] = 0.9  # Default for nailed starters

    result["EP_app"] = result["_p60"].apply(lambda p: 2.0 if p >= 0.67 else 1.0 * p / 0.67)

    # ── EP_att: Attacking points ──
    result["_goal_pts"] = result["position"].map(GOAL_PTS).fillna(4)

    # Use npxG if player isn't primary penalty taker (penalty_order > 1 or unavailable)
    if npxg_col and npxg_col in result.columns:
        penalty_order = result.get("penalty_order", pd.Series([99] * len(result)))
        penalty_order = pd.to_numeric(penalty_order, errors="coerce").fillna(99)
        result["_xg_adj"] = result.apply(
            lambda r: r[npxg_col] if penalty_order.get(r.name, 99) > 1 else r[xg_col], axis=1
        )
    else:
        result["_xg_adj"] = result[xg_col]

    # Poisson-adjusted goal expectation: sum over k of k * P(k goals)
    # Approx: xG * 1.05 for small xG captures haul variance
    result["EP_att"] = (
        result["_xg_adj"] * result["_goal_pts"] * 1.05  # slight haul boost
        + result[xa_col] * ASSIST_PTS
    )

    # ── EP_def: Defensive points ──
    cs_col = "clean_sheets_per_90" if "clean_sheets_per_90" in result.columns else None
    xgc_col = "expected_goals_conceded_per_90" if "expected_goals_conceded_per_90" in result.columns else None

    result["_cs_pts"] = result["position"].map(CS_PTS).fillna(0)
    if cs_col and cs_col in result.columns:
        result[cs_col] = pd.to_numeric(result[cs_col], errors="coerce").fillna(0)
        result["_p_cs"] = result[cs_col]  # Per-90 CS rate as probability proxy
    else:
        result["_p_cs"] = 0.0

    # DefCon bonus (probability of reaching threshold)
    cbit_prop_col = "cbit_propensity" if "cbit_propensity" in result.columns else None
    if cbit_prop_col:
        result[cbit_prop_col] = pd.to_numeric(result[cbit_prop_col], errors="coerce").fillna(0)
        result["_defcon"] = result[cbit_prop_col] * DEFCON_BONUS
    else:
        result["_defcon"] = 0.0

    # xGC penalty: -1 pt per 2 xGC for GKP/DEF
    if xgc_col and xgc_col in result.columns:
        result[xgc_col] = pd.to_numeric(result[xgc_col], errors="coerce").fillna(0)
        result["_xgc_penalty"] = result.apply(
            lambda r: -(r[xgc_col] / 2) if r["position"] in ("GKP", "DEF") else 0, axis=1
        )
    else:
        result["_xgc_penalty"] = 0.0

    result["EP_def"] = result["_p_cs"] * result["_cs_pts"] + result["_defcon"] + result["_xgc_penalty"]

    # ── EP_bonus: Bonus expectation ~~0.10 * xGI ──
    result["_xgi"] = result["_xg_adj"] + result[xa_col]
    result["EP_bonus"] = result["_xgi"] * 0.10

    # ── Total EP (single gameweek) ──
    result["EP_advanced"] = (
        result["EP_app"] + result["EP_att"] + result["EP_def"] + result["EP_bonus"]
    ).clip(lower=0)

    # ── Blend with FPL base EP (50/50) for stability ──
    base_ep = pd.to_numeric(result.get("ep_next", result.get("expected_points", 0)), errors="coerce").fillna(0)
    result["expected_points"] = (result["EP_advanced"] * 0.6 + base_ep * 0.4).clip(lower=0)

    # ── Home advantage boost (10% if fixture data available) ──
    if "is_home" in result.columns:
        home_mask = result["is_home"] == True
        result.loc[home_mask, "expected_points"] *= 1.10

    # ── Cleanup temp columns ──
    temp_cols = [
        "_games_est", "_p60", "_goal_pts", "_xg_adj", "_cs_pts", "_p_cs",
        "_defcon", "_xgc_penalty", "_xgi"
    ]
    result.drop(columns=[c for c in temp_cols if c in result.columns], inplace=True, errors="ignore")

    return result


def adjust_ep_with_understat(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper: calculates advanced EP then applies regression adjustment."""
    result = calculate_advanced_ep(df)
    return apply_regression_adjustment(result)


def apply_regression_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Apply regression signals (over/underperformance) to fine-tune EP.

    - Goals > xG (overperforming) -> slight EP decrease (regression risk)
    - Goals < xG (underperforming) -> slight EP increase (bounce-back)
    - Same for assists vs xA
    - Adjustment capped at +/-15% of EP
    """
    if "us_goal_overperf" not in df.columns or "expected_points" not in df.columns:
        return df

    result = df.copy()
    has_data = result.get("us_xG", pd.Series([0] * len(result))) > 0
    result["_reg_adj"] = 0.0

    for pos, (gw, aw, bw) in POS_WEIGHTS.items():
        mask = has_data & (result["position"] == pos)
        if not mask.any():
            continue

        # Negative overperf -> underperforming -> positive adjustment
        goal_adj = -result.loc[mask, "us_goal_overperf"] * gw * 0.5
        assist_adj = -result.loc[mask, "us_assist_overperf"] * aw * 0.5
        buildup = result.loc[mask, "us_xGBuildup_per90"] * bw * 5

        raw = goal_adj + assist_adj + buildup
        cap = result.loc[mask, "expected_points"].clip(lower=0.1) * 0.15
        result.loc[mask, "_reg_adj"] = raw.clip(lower=-cap, upper=cap)

    result["expected_points"] = (result["expected_points"] + result["_reg_adj"]).clip(lower=0)
    result.drop(columns=["_reg_adj"], inplace=True)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_with_understat(fpl_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch Understat data, match to FPL players, adjust EP.

    Caches both player and **team** Understat data in ``st.session_state``
    so repeated calls within a session do not trigger additional HTTP
    requests.  Team-level stats (xG-for, xGA) are stored at
    ``st.session_state['_understat_team_stats']`` for the Poisson engine.

    Stores detailed data-source status in ``st.session_state['_understat_status']``
    so downstream consumers (Poisson engine, diagnostics) can inspect
    exactly what succeeded and what fell back.
    """
    import time
    import traceback
    import streamlit as st

    cache_players = "_understat_raw"
    cache_teams = "_understat_teams_raw"
    cache_team_stats = "_understat_team_stats"

    status: dict = {
        "players_fetched": 0,
        "teams_fetched": 0,
        "team_stats_built": False,
        "team_stats_count": 0,
        "player_match_rate": 0.0,
        "errors": [],
    }

    # ── Fetch or use cached data ──
    if not force_refresh and cache_players in st.session_state:
        raw = st.session_state[cache_players]
        teams_raw = st.session_state.get(cache_teams, {})
        logger.debug("Using cached Understat data (%d players)", len(raw))
    else:
        t0 = time.time()
        try:
            raw, teams_raw = fetch_understat_league_data()
        except Exception as exc:
            logger.error("Understat fetch raised unexpected error: %s\n%s",
                         exc, traceback.format_exc())
            status["errors"].append(f"fetch: {exc}")
            raw, teams_raw = [], {}
        elapsed = time.time() - t0
        logger.info("Understat fetch completed in %.1fs (%d players, %d teams)",
                     elapsed, len(raw), len(teams_raw))
        st.session_state[cache_players] = raw
        st.session_state[cache_teams] = teams_raw

    status["players_fetched"] = len(raw)
    status["teams_fetched"] = len(teams_raw)

    if not raw:
        st.session_state["_understat_active"] = False
        status["errors"].append("No player data returned from Understat")
        st.session_state["_understat_status"] = status
        logger.warning("Understat enrichment skipped: no player data available")
        return fpl_df

    st.session_state["_understat_active"] = True

    # ── Build and cache team-level stats for the Poisson engine ──
    # Rebuild if empty DataFrame was cached from a previous partial failure
    existing_team_stats = st.session_state.get(cache_team_stats)
    need_rebuild = (
        existing_team_stats is None
        or (isinstance(existing_team_stats, pd.DataFrame) and existing_team_stats.empty)
    )
    if teams_raw and (force_refresh or need_rebuild):
        try:
            team_stats_df = build_team_stats_df(teams_raw)
            st.session_state[cache_team_stats] = team_stats_df
            status["team_stats_built"] = not team_stats_df.empty
            status["team_stats_count"] = len(team_stats_df)
            if team_stats_df.empty:
                logger.warning("build_team_stats_df returned empty DataFrame "
                               "despite %d raw teams", len(teams_raw))
                status["errors"].append("team_stats_df empty after build")
        except Exception as exc:
            logger.error("Failed to build team stats from Understat: %s\n%s",
                         exc, traceback.format_exc())
            status["errors"].append(f"team_stats_build: {exc}")
            st.session_state[cache_team_stats] = pd.DataFrame()
    elif not teams_raw and need_rebuild:
        logger.warning("No Understat team data available; "
                       "Poisson engine will use FDR fallback for all teams")
        status["errors"].append("No team data from Understat")
        st.session_state[cache_team_stats] = pd.DataFrame()
    else:
        # Using previously cached team stats
        cached = st.session_state.get(cache_team_stats, pd.DataFrame())
        status["team_stats_built"] = isinstance(cached, pd.DataFrame) and not cached.empty
        status["team_stats_count"] = len(cached) if isinstance(cached, pd.DataFrame) else 0

    # ── Player enrichment ──
    try:
        ustat_df = _build_understat_df(raw)
        merged = match_understat_to_fpl(fpl_df, ustat_df)

        # Track match quality
        if "us_xG" in merged.columns:
            matched = (merged["us_xG"].fillna(0) > 0).sum()
            status["player_match_rate"] = round(matched / max(len(merged), 1), 3)
            logger.info("Understat player match rate: %d / %d (%.0f%%)",
                        matched, len(merged),
                        status["player_match_rate"] * 100)

        adjusted = adjust_ep_with_understat(merged)
    except Exception as exc:
        logger.error("Understat player enrichment failed: %s\n%s",
                     exc, traceback.format_exc())
        status["errors"].append(f"enrichment: {exc}")
        adjusted = fpl_df

    st.session_state["_understat_status"] = status
    return adjusted
