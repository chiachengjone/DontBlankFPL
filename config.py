"""
Centralized configuration for FPL Strategy Engine.
All magic numbers, thresholds, and season constants live here.
"""

from typing import Dict

# ══════════════════════════════════════════════════════════════════════════════
# 2025/26 SEASON RULES
# ══════════════════════════════════════════════════════════════════════════════

SEASON: str = "2025/26"
MAX_FREE_TRANSFERS: int = 5
CAPTAIN_MULTIPLIER: float = 1.25
CBIT_BONUS_THRESHOLD: int = 10
CBIT_BONUS_POINTS: int = 2

# ══════════════════════════════════════════════════════════════════════════════
# SQUAD CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════

MAX_BUDGET: float = 100.0
MAX_PLAYERS_PER_TEAM: int = 3
SQUAD_SIZE: int = 15
STARTING_XI: int = 11

POSITION_CONSTRAINTS: Dict[str, Dict[str, int]] = {
    'GKP': {'min': 2, 'max': 2, 'min_start': 1, 'max_start': 1},
    'DEF': {'min': 5, 'max': 5, 'min_start': 3, 'max_start': 5},
    'MID': {'min': 5, 'max': 5, 'min_start': 2, 'max_start': 5},
    'FWD': {'min': 3, 'max': 3, 'min_start': 1, 'max_start': 3},
}

# ══════════════════════════════════════════════════════════════════════════════
# FPL 2025/26 SCORING RULES
# ══════════════════════════════════════════════════════════════════════════════

GOAL_POINTS: Dict[str, int] = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS: int = 3
CLEAN_SHEET_POINTS: Dict[str, int] = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
APPEARANCE_POINTS: Dict[int, int] = {60: 2, 1: 1}  # 60+ mins → 2pts, 1-59 → 1pt
CBIT_THRESHOLDS: Dict[str, int] = {"GKP": 12, "DEF": 10, "MID": 12, "FWD": 12}
GOALS_CONCEDED_PER_2_PENALTY: int = -1  # GKP/DEF lose 1pt per 2 goals conceded
SAVES_PER_BONUS_POINT: int = 3  # 3 saves = 1pt (goalkeepers)
BONUS_POINT_POSITION_MULTIPLIER: Dict[str, float] = {
    "GKP": 1.5, "DEF": 1.3, "MID": 1.0, "FWD": 0.85,
}

# ══════════════════════════════════════════════════════════════════════════════
# HOME / AWAY MULTIPLIERS (based on historical EPL data)
# ══════════════════════════════════════════════════════════════════════════════

HOME_ADVANTAGE_ATTACK: float = 1.15   # ~15% more goals at home
HOME_ADVANTAGE_DEFENSE: float = 0.90  # ~10% fewer goals conceded at home
AWAY_MULTIPLIER_ATTACK: float = 0.87
AWAY_MULTIPLIER_DEFENSE: float = 1.10

# ══════════════════════════════════════════════════════════════════════════════
# LEAGUE AVERAGES (updated each season)
# ══════════════════════════════════════════════════════════════════════════════

LEAGUE_AVG_XG_PER_MATCH: float = 1.35   # ~2.7 goals/game split between teams
LEAGUE_AVG_XGA_PER_MATCH: float = 1.35

# ══════════════════════════════════════════════════════════════════════════════
# POISSON CALCULATION LIMITS
# ══════════════════════════════════════════════════════════════════════════════

MAX_K_GOALS: int = 5   # Player rarely scores 5+ in a match
MAX_K_ASSISTS: int = 4
CLEAN_SHEET_PROBABILITY_CAP: float = 0.60  # Even best teams rarely > 50%

# ══════════════════════════════════════════════════════════════════════════════
# MODEL BLEND WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_WEIGHTS: Dict[str, float] = {
    'ml': 0.4,
    'poisson': 0.4,
    'fpl': 0.2,
}
ML_FALLBACK_RATIO: float = 0.9  # When no ML data, use poisson × this

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEEKS_AHEAD: int = 5
DECAY_FACTOR: float = 0.95
TRANSFER_HIT_COST: int = 4

# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

MC_DEFAULT_SIMULATIONS: int = 10_000

# ══════════════════════════════════════════════════════════════════════════════
# FDR (Fixture Difficulty Rating) THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

FDR_EASY: int = 2       # <= 2 is easy
FDR_MEDIUM: int = 3     # 3 is medium
FDR_HARD: int = 4       # >= 4 is tough

FDR_COLOR_MAP: Dict[int, str] = {
    1: '#22c55e',
    2: '#77c45e',
    3: '#f59e0b',
    4: '#ef6b4e',
    5: '#ef4444',
}

# ══════════════════════════════════════════════════════════════════════════════
# OWNERSHIP TIERS
# ══════════════════════════════════════════════════════════════════════════════

OWNERSHIP_TEMPLATE_THRESHOLD: float = 25.0   # >= 25% = template
OWNERSHIP_POPULAR_THRESHOLD: float = 10.0    # 10-25% = popular
OWNERSHIP_DIFFERENTIAL_THRESHOLD: float = 5.0  # < 5% = differential
# 5-10% = enabler

# ══════════════════════════════════════════════════════════════════════════════
# ROTATION RISK THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

ROTATION_SAFE_MINUTES_PCT: float = 0.80    # >= 80% available mins = safe
ROTATION_MODERATE_PCT: float = 0.60        # 60-80% = moderate risk
# < 60% = high risk

# ══════════════════════════════════════════════════════════════════════════════
# INJURY / AVAILABILITY STATUS CODES
# ══════════════════════════════════════════════════════════════════════════════

STATUS_AVAILABLE: str = 'a'
STATUS_INJURED: str = 'i'
STATUS_DOUBTFUL: str = 'd'
STATUS_SUSPENDED: str = 's'
STATUS_UNAVAILABLE: str = 'u'
UNAVAILABLE_STATUSES: set = {'i', 's', 'u'}

# ══════════════════════════════════════════════════════════════════════════════
# PRICE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

PRICE_RISE_NET_TRANSFERS: int = 50_000
PRICE_FALL_NET_TRANSFERS: int = -50_000

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD QUICK PICK LIMITS
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_TOP_PICKS: int = 5
DASHBOARD_INJURY_ALERTS: int = 8
DASHBOARD_FIXTURE_SWINGS: int = 6

# ══════════════════════════════════════════════════════════════════════════════
# CHART THEME
# ══════════════════════════════════════════════════════════════════════════════

CHART_BG: str = '#ffffff'
CHART_PLOT_BG: str = '#ffffff'
CHART_GRID: str = '#e5e5ea'
CHART_FONT_FAMILY: str = 'Inter, sans-serif'
CHART_FONT_COLOR: str = '#86868b'
CHART_FONT_SIZE: int = 11

POSITION_COLORS: Dict[str, str] = {
    'GKP': '#3b82f6',
    'DEF': '#22c55e',
    'MID': '#f59e0b',
    'FWD': '#ef4444',
}

# ══════════════════════════════════════════════════════════════════════════════
# API CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FPL_BASE_URL: str = "https://fantasy.premierleague.com/api"
ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
CACHE_DURATION: int = 300  # 5 minutes


def get_chart_layout(height: int = 400, **overrides) -> dict:
    """Return a standard Plotly layout dict for consistent styling."""
    layout = dict(
        height=height,
        template='plotly_white',
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_PLOT_BG,
        font=dict(family=CHART_FONT_FAMILY, color=CHART_FONT_COLOR, size=CHART_FONT_SIZE),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
    )
    layout.update(overrides)
    return layout
