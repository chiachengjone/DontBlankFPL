"""
Centralized configuration for FPL Strategy Engine.
All magic numbers, thresholds, and season constants live here.
"""

# ── 2025/26 Season Rules ──
SEASON = "2025/26"
MAX_FREE_TRANSFERS = 5
CAPTAIN_MULTIPLIER = 1.25
CBIT_BONUS_THRESHOLD = 10
CBIT_BONUS_POINTS = 2

# ── Squad Constraints ──
MAX_BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3
SQUAD_SIZE = 15
STARTING_XI = 11
POSITION_CONSTRAINTS = {
    'GKP': {'min': 2, 'max': 2, 'min_start': 1, 'max_start': 1},
    'DEF': {'min': 5, 'max': 5, 'min_start': 3, 'max_start': 5},
    'MID': {'min': 5, 'max': 5, 'min_start': 2, 'max_start': 5},
    'FWD': {'min': 3, 'max': 3, 'min_start': 1, 'max_start': 3},
}

# ── Optimization Defaults ──
DEFAULT_WEEKS_AHEAD = 5
DECAY_FACTOR = 0.95
TRANSFER_HIT_COST = 4

<<<<<<< Updated upstream
# ── FDR Thresholds ──
FDR_EASY = 2       # <= 2 is easy
FDR_MEDIUM = 3     # 3 is medium
FDR_HARD = 4       # >= 4 is tough
FDR_COLOR_MAP = {
=======
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

# Monte Carlo variance estimation weights
# estimated_std = ep * MC_STD_EP_WEIGHT + (MC_STD_FORM_BASELINE - form) * MC_STD_FORM_WEIGHT
#                 + rotation_penalty
MC_STD_EP_WEIGHT: float = 0.4       # Base volatility proportional to EP
MC_STD_FORM_BASELINE: float = 5.0   # Form value treated as "average"
MC_STD_FORM_WEIGHT: float = 0.3     # Poor form adds this per unit below baseline
MC_STD_ROTATION_WEIGHT: float = 2.0 # Full-rotation player gets +2 std
MC_STD_MIN: float = 0.5             # Floor on estimated std
MC_EPSILON: float = 0.1             # Minimum EP/std for numerical safety

# ML prediction uncertainty parameters
ML_BASE_UNCERTAINTY: float = 0.5    # Minimum inherent prediction uncertainty
ML_LOW_EP_UNCERTAINTY: float = 0.3  # Extra uncertainty for low-EP players
ML_LOW_EP_THRESHOLD: float = 4.0    # EP below which extra uncertainty applies

# Genetic optimizer scoring weights
GA_DIVERSITY_BONUS: float = 0.5     # Points per unique team in squad
GA_BUDGET_BONUS: float = 0.1        # Fraction of remaining budget added to fitness
GA_MIN_FITNESS: float = -500.0      # Floor on fitness to prevent numerical issues

# ══════════════════════════════════════════════════════════════════════════════
# BENCH STRENGTH & RISK-AWARE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Weight given to bench xP in the optimizer objective (0 = ignore bench)
BENCH_XP_WEIGHT: float = 0.15

# Risk-mode multiplier: how much variance (std) to subtract from xP
# risk_score = xP - RISK_AVERSION * std(xP)
# Higher = more conservative; use 0 for pure xP maximisation
RISK_AVERSION_DEFAULT: float = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODEL (BIVARIATE POISSON)
# ══════════════════════════════════════════════════════════════════════════════

# Intra-team attack correlation coefficient (goals ↔ assists on same team)
TEAM_ATTACK_CORRELATION: float = 0.20

# ══════════════════════════════════════════════════════════════════════════════
# INJURY RECOVERY CURVES (by keyword category)
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (fixture_0_multiplier, fixture_1_mult, fixture_2_mult, fixture_3+_mult)
INJURY_CURVES: Dict[str, tuple] = {
    # Tactical knock / illness – fast recovery
    "knock":     (0.60, 0.90, 1.00, 1.00),
    "illness":   (0.50, 0.85, 1.00, 1.00),
    # Muscle strain – moderate recovery
    "hamstring":  (0.10, 0.40, 0.75, 0.95),
    "groin":      (0.10, 0.40, 0.70, 0.90),
    "calf":       (0.10, 0.45, 0.75, 0.95),
    "muscle":     (0.15, 0.45, 0.75, 0.95),
    "thigh":      (0.10, 0.40, 0.70, 0.90),
    # Joint / ligament – slow recovery
    "ankle":      (0.05, 0.20, 0.50, 0.80),
    "knee":       (0.05, 0.15, 0.40, 0.70),
    "foot":       (0.05, 0.25, 0.55, 0.80),
    "back":       (0.10, 0.30, 0.60, 0.85),
    "shoulder":   (0.10, 0.35, 0.65, 0.85),
    # Severe / long-term – effectively unavailable for the horizon
    "acl":        (0.00, 0.00, 0.00, 0.05),
    "surgery":    (0.00, 0.00, 0.00, 0.05),
    "months":     (0.00, 0.00, 0.00, 0.10),
    "season":     (0.00, 0.00, 0.00, 0.00),
    "indefinite": (0.00, 0.00, 0.00, 0.05),
    # Default (unrecognised news text)
    "default":    (0.40, 0.70, 0.90, 1.00),
}

# ══════════════════════════════════════════════════════════════════════════════
# CBIT DECAY WEIGHTS (recency-biased estimation)
# ══════════════════════════════════════════════════════════════════════════════

# Exponential decay half-life in gameweeks for CBIT estimation
CBIT_DECAY_HALFLIFE_GW: int = 6

# ══════════════════════════════════════════════════════════════════════════════
# FDR (Fixture Difficulty Rating) THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

FDR_EASY: int = 2       # <= 2 is easy
FDR_MEDIUM: int = 3     # 3 is medium
FDR_HARD: int = 4       # >= 4 is tough

FDR_COLOR_MAP: Dict[int, str] = {
>>>>>>> Stashed changes
    1: '#22c55e',
    2: '#77c45e',
    3: '#f59e0b',
    4: '#ef6b4e',
    5: '#ef4444',
}

# ── Ownership Tiers ──
OWNERSHIP_TEMPLATE_THRESHOLD = 25.0   # >= 25% = template
OWNERSHIP_POPULAR_THRESHOLD = 10.0    # 10-25% = popular
OWNERSHIP_DIFFERENTIAL_THRESHOLD = 5.0  # < 5% = differential
# 5-10% = enabler

# ── Rotation Risk Thresholds ──
ROTATION_SAFE_MINUTES_PCT = 0.80      # >= 80% available mins = safe
ROTATION_MODERATE_PCT = 0.60          # 60-80% = moderate risk
# < 60% = high risk

# ── Injury/Availability Status Codes ──
STATUS_AVAILABLE = 'a'
STATUS_INJURED = 'i'
STATUS_DOUBTFUL = 'd'
STATUS_SUSPENDED = 's'
STATUS_UNAVAILABLE = 'u'
UNAVAILABLE_STATUSES = {'i', 's', 'u'}

# ── Price Thresholds ──
PRICE_RISE_NET_TRANSFERS = 50_000
PRICE_FALL_NET_TRANSFERS = -50_000

# ── Dashboard Quick Pick Limits ──
DASHBOARD_TOP_PICKS = 5
DASHBOARD_INJURY_ALERTS = 8
DASHBOARD_FIXTURE_SWINGS = 6

# ── Chart Theme ──
CHART_BG = '#ffffff'
CHART_PLOT_BG = '#ffffff'
CHART_GRID = '#e5e5ea'
CHART_FONT_FAMILY = 'Inter, sans-serif'
CHART_FONT_COLOR = '#86868b'
CHART_FONT_SIZE = 11

POSITION_COLORS = {
    'GKP': '#3b82f6',
    'DEF': '#22c55e',
    'MID': '#f59e0b',
    'FWD': '#ef4444',
}

<<<<<<< Updated upstream
# ── API Config ──
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_DURATION = 300  # 5 minutes
=======
# ══════════════════════════════════════════════════════════════════════════════
# API CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FPL_BASE_URL: str = "https://fantasy.premierleague.com/api"
CACHE_DURATION: int = 300  # 5 minutes
>>>>>>> Stashed changes


def get_chart_layout(height=400, **overrides):
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
